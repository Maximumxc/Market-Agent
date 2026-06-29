"""
us_close_report.py — MFTSR Alpha · 美股收盘完整报告（v5）

v5 更新（对齐 MFTSR_AI_Growth_Model_EN.xlsx 精确框架）：
  1. 权重改为与Excel模型完全一致：宏观20% / 基本面35% / 技术面20% / 情绪10% / 风险15%
  2. 每个维度内部子指标权重与打分规则严格对应Excel（见各 score_* 函数注释）
  3. AI Industrial Policy / Geopolitical Score / CNN Fear & Greed / Market Liquidity
     四项已从MFTSR评分中移除（按用户指示），其定性信息仅作为宏观简报的文字背景，
     不计入任何维度评分。
  4. MFTSR评分区间改为6档（与Excel Dashboard的Action guide一致）：
     ≥85 积极加仓 / 75-85 买入 / 60-75 持有逢低买入 / 50-60 谨慎 / 40-50 减仓 / <40 大幅减仓
  5. AI解读改为5个独立段落（基本面/技术面/情绪/风险/宏观影响各一段），
     每段单独标注该维度评分。
  6. 频次改为每周一、三、五（原每个交易日），UK时间21:30不变。
  7. 新增 export_dashboard_json()：把当次真实价格/评分/AI解读导出为JSON，
     提交到仓库 docs/ 目录，供 GitHub Pages 托管的Dashboard网页读取。

环境变量：
    ANTHROPIC_KEY   — Claude API key
    TELEGRAM_TOKEN  — Telegram bot token
    CHAT_ID         — Telegram chat id
    FRED_API_KEY    — (可选) 用于CPI/非农/失业率/Fed利率官方数据
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

import requests
import numpy as np
import yfinance as yf
import anthropic

from shared_config import WEIGHTS, US_WATCHLIST as DEFAULT_WATCHLIST

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mftsr")

ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
FRED_API_KEY   = os.environ.get("FRED_API_KEY", "")
# 控制是否发送Telegram消息。设为"false"时，脚本仍会完整计算所有评分+AI解读+
# 监测模块，并导出Dashboard JSON，但跳过所有Telegram发送——用于"只看网页，
# 不再接收推送"的场景（仍保留每周自动运行以刷新网页数据）。
SEND_TELEGRAM = os.environ.get("SEND_TELEGRAM", "true").lower() not in ("false", "0", "no")

_watchlist_env = os.environ.get("WATCHLIST", "")
if _watchlist_env.strip():
    WATCHLIST = [{"sym": s.strip().upper(), "name": s.strip().upper(), "sector": "Custom"}
                 for s in _watchlist_env.split(",") if s.strip()]
else:
    WATCHLIST = DEFAULT_WATCHLIST

SCORE_BANDS = [
    (85, 101, "强烈买入", "积极加仓", "🟢🟢"),
    (75, 85,  "买入",     "买入",     "🟢"),
    (60, 75,  "持有偏多", "持有/逢低买入", "🟡"),
    (50, 60,  "谨慎",     "谨慎",     "🟠"),
    (40, 50,  "减仓",     "减仓",     "🔴"),
    (0,  40,  "强烈卖出", "大幅减仓", "🔴🔴"),
]
ALERT_DIM = 30


def score_band(score: int) -> tuple:
    for lo, hi, label, action, emoji in SCORE_BANDS:
        if lo <= score < hi:
            return label, action, emoji
    return SCORE_BANDS[-1][2], SCORE_BANDS[-1][3], SCORE_BANDS[-1][4]


SCORE_LEGEND_ZH = (
    "📐 <b>MFTSR评分标准</b>（与Excel模型Action Guide一致）\n"
    "  🟢🟢 ≥85分    强烈买入 → 积极加仓\n"
    "  🟢   75-85分  买入    → 买入\n"
    "  🟡   60-75分  持有偏多 → 持有/逢低买入\n"
    "  🟠   50-60分  谨慎    → 谨慎\n"
    "  🔴   40-50分  减仓    → 减仓\n"
    "  🔴🔴 &lt;40分    强烈卖出 → 大幅减仓"
)


# ════════════════════════════════════════════════════════════════════════════
#  1. 市场数据抓取
# ════════════════════════════════════════════════════════════════════════════

def get_market_data(sym: str) -> dict:
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="90d", interval="1d")
        if hist.empty:
            return _fallback(sym)

        close, volume = hist["Close"], hist["Volume"]
        n_days = len(close)

        # ⚠️ 关键修复：yfinance偶尔返回'非空但最后一行是NaN'的数据（当天数据
        # 未完全同步），hist.empty检查通不过但latest会变成nan，后续'is not None'
        # 判断拦不住nan（nan确实不是None），导致价格显示异常。这里显式拦截。
        if np.isnan(close.iloc[-1]) or close.iloc[-1] <= 0:
            logger.warning(f"{sym} 最新价格为NaN或非正数，视为数据不可用")
            return _fallback(sym)

        latest = float(close.iloc[-1])
        prev   = float(close.iloc[-2]) if len(close) > 1 and not np.isnan(close.iloc[-2]) else latest
        change_pct = (latest - prev) / prev * 100

        hi_52w = float(close.tail(252).max())
        lo_52w = float(close.tail(252).min())

        def _safe_ma(rolling_mean_series):
            val = rolling_mean_series.iloc[-1]
            return float(val) if not np.isnan(val) else None

        ma5   = _safe_ma(close.rolling(5).mean())   if n_days >= 5   else None
        ma20  = _safe_ma(close.rolling(20).mean())  if n_days >= 20  else None
        ma50  = _safe_ma(close.rolling(50).mean())  if n_days >= 50  else None
        ma200 = _safe_ma(close.rolling(200).mean()) if n_days >= 200 else None
        thin_history = n_days < 50

        ma_cross = None
        if n_days >= 21:
            ma5_series  = close.rolling(5).mean()
            ma20_series = close.rolling(20).mean()
            today_diff  = ma5_series.iloc[-1] - ma20_series.iloc[-1]
            yest_diff   = ma5_series.iloc[-2] - ma20_series.iloc[-2]
            if not (np.isnan(today_diff) or np.isnan(yest_diff)):
                if yest_diff <= 0 and today_diff > 0:
                    ma_cross = "golden"
                elif yest_diff >= 0 and today_diff < 0:
                    ma_cross = "death"

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float(100 - (100 / (1 + rs)).iloc[-1]) if n_days >= 14 and not rs.isna().iloc[-1] else None

        avg_vol_raw = float(volume.rolling(20).mean().iloc[-1]) if n_days >= 20 else float(volume.mean())
        avg_vol = avg_vol_raw if not np.isnan(avg_vol_raw) else 0
        latest_vol = float(volume.iloc[-1])
        latest_vol = latest_vol if not np.isnan(latest_vol) else 0
        vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0

        rel_strength_3m = None
        try:
            qqq_hist = yf.Ticker("QQQ").history(period="90d", interval="1d")
            if len(qqq_hist) >= 60 and n_days >= 60:
                stock_3m_ret = (close.iloc[-1] / close.iloc[-60] - 1) * 100
                qqq_3m_ret = (qqq_hist["Close"].iloc[-1] / qqq_hist["Close"].iloc[-60] - 1) * 100
                rel_strength_3m = float(stock_3m_ret - qqq_3m_ret)
        except Exception:
            pass

        vol_30d_annualized = None
        if n_days >= 31:
            daily_returns = close.pct_change().dropna().tail(30)
            vol_30d_annualized = float(daily_returns.std() * np.sqrt(252) * 100)

        info = ticker.info
        pe_ratio      = info.get("trailingPE") or 0
        forward_pe    = info.get("forwardPE") or 0
        peg_ratio     = info.get("pegRatio") or info.get("trailingPegRatio") or 0
        rev_growth    = (info.get("revenueGrowth") or 0) * 100
        eps_growth    = (info.get("earningsGrowth") or 0) * 100
        gross_margin  = (info.get("grossMargins") or 0) * 100
        fcf_yield     = 0
        if info.get("freeCashflow") and info.get("marketCap"):
            fcf_yield = info["freeCashflow"] / info["marketCap"] * 100
        fcf_margin = 0
        if info.get("freeCashflow") and info.get("totalRevenue"):
            fcf_margin = info["freeCashflow"] / info["totalRevenue"] * 100
        short_ratio   = info.get("shortRatio") or 0
        inst_pct      = (info.get("heldPercentInstitutions") or 0) * 100
        market_cap_b  = (info.get("marketCap") or 0) / 1e9
        beta          = info.get("beta") or 1.0
        analyst_rec   = info.get("recommendationMean") or 3.0
        target_price  = info.get("targetMeanPrice") or latest
        target_high   = info.get("targetHighPrice") or target_price
        target_low    = info.get("targetLowPrice") or target_price
        num_analysts  = info.get("numberOfAnalystOpinions") or 0
        upside        = (target_price - latest) / latest * 100 if latest > 0 else 0

        rev_cagr_proxy = rev_growth
        profit_cagr_proxy = eps_growth

        return {
            "sym": sym, "price": round(latest, 2), "change_pct": round(change_pct, 2),
            "hi_52w": round(hi_52w, 2), "lo_52w": round(lo_52w, 2),
            "pct_from_52w_hi": round((latest / hi_52w - 1) * 100, 1) if hi_52w else 0,
            "ma5": round(ma5, 2) if ma5 is not None else "N/A",
            "ma20": round(ma20, 2) if ma20 is not None else "N/A",
            "ma50": round(ma50, 2) if ma50 is not None else "N/A",
            "ma200": round(ma200, 2) if ma200 is not None else "N/A",
            "above_ma20": (latest > ma20) if ma20 is not None else None,
            "above_ma50": (latest > ma50) if ma50 is not None else None,
            "above_ma200": (latest > ma200) if ma200 is not None else None,
            "ma_cross": ma_cross,
            "rsi": round(rsi, 1) if rsi is not None else "N/A",
            "vol_ratio": round(vol_ratio, 2),
            "rel_strength_3m": round(rel_strength_3m, 1) if rel_strength_3m is not None else "N/A",
            "vol_30d_annualized": round(vol_30d_annualized, 1) if vol_30d_annualized is not None else "N/A",
            "pe_ratio": round(pe_ratio, 1) if pe_ratio else "N/A",
            "forward_pe": round(forward_pe, 1) if forward_pe else "N/A",
            "peg_ratio": round(peg_ratio, 2) if peg_ratio else "N/A",
            "rev_growth": round(rev_growth, 1), "eps_growth": round(eps_growth, 1),
            "rev_cagr_proxy": round(rev_cagr_proxy, 1), "profit_cagr_proxy": round(profit_cagr_proxy, 1),
            "gross_margin": round(gross_margin, 1), "fcf_yield": round(fcf_yield, 1),
            "fcf_margin": round(fcf_margin, 1),
            "market_cap_b": round(market_cap_b, 1),
            "beta": round(beta, 2), "short_ratio": round(short_ratio, 1),
            "inst_pct": round(inst_pct, 1), "analyst_rec": round(analyst_rec, 2),
            "target_price": round(target_price, 2), "target_high": round(target_high, 2),
            "target_low": round(target_low, 2), "num_analysts": num_analysts,
            "upside": round(upside, 1),
            "thin_history": thin_history, "trading_days": n_days,
        }
    except Exception as e:
        logger.error(f"{sym} 数据抓取失败: {e}")
        return _fallback(sym)


def _fallback(sym: str) -> dict:
    return {
        "sym": sym, "price": 0, "change_pct": 0, "hi_52w": 0, "lo_52w": 0, "pct_from_52w_hi": 0,
        "ma5": "N/A", "ma20": "N/A", "ma50": "N/A", "ma200": "N/A",
        "above_ma20": None, "above_ma50": None, "above_ma200": None, "ma_cross": None,
        "rsi": "N/A", "vol_ratio": 1, "rel_strength_3m": "N/A", "vol_30d_annualized": "N/A",
        "pe_ratio": "N/A", "forward_pe": "N/A", "peg_ratio": "N/A",
        "rev_growth": 0, "eps_growth": 0, "rev_cagr_proxy": 0, "profit_cagr_proxy": 0,
        "gross_margin": 0, "fcf_yield": 0, "fcf_margin": 0,
        "market_cap_b": 0, "beta": 1.0, "short_ratio": 0, "inst_pct": 0,
        "analyst_rec": 3, "target_price": 0, "target_high": 0, "target_low": 0,
        "num_analysts": 0, "upside": 0,
        "thin_history": True, "trading_days": 0, "error": True,
    }


def get_macro_market_snapshot() -> dict:
    tickers_map = {
        "vix": "^VIX", "dxy": "DX-Y.NYB", "us10y": "^TNX", "us2y": "^IRX",
        "crude": "CL=F", "gold": "GC=F", "sp500": "^GSPC", "nasdaq": "^IXIC",
    }
    result = {}
    for key, sym in tickers_map.items():
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                latest = float(h["Close"].iloc[-1])
                prev   = float(h["Close"].iloc[-2]) if len(h) > 1 else latest
                result[key] = round(latest, 2)
                result[f"{key}_chg_pct"] = round((latest - prev) / prev * 100, 2) if prev else 0
        except Exception:
            result[key] = None
    return result


FRED_SERIES = {
    "cpi_yoy": "CPIAUCSL", "core_pce_yoy": "PCEPILFE",
    "nfp": "PAYEMS", "unemployment": "UNRATE", "fed_funds": "FEDFUNDS",
}


def _fred_latest_two(series_id: str) -> tuple:
    if not FRED_API_KEY:
        return None, None, None
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": FRED_API_KEY,
                   "file_type": "json", "sort_order": "desc", "limit": 14}
        resp = requests.get(url, params=params, timeout=10)
        obs = [o for o in resp.json().get("observations", []) if o["value"] not in (".", "")]
        if len(obs) < 2:
            return None, None, None
        return float(obs[0]["value"]), float(obs[1]["value"]), obs[0]["date"]
    except Exception as e:
        logger.warning(f"FRED获取失败 {series_id}: {e}")
        return None, None, None


def get_macro_economic_data() -> dict:
    if not FRED_API_KEY:
        return {}
    econ = {}
    try:
        cpi_now, cpi_prev, _ = _fred_latest_two(FRED_SERIES["cpi_yoy"])
        if cpi_now and cpi_prev:
            econ["cpi_mom_pct"] = round((cpi_now - cpi_prev) / cpi_prev * 100, 2)
        pce_now, pce_prev, _ = _fred_latest_two(FRED_SERIES["core_pce_yoy"])
        if pce_now and pce_prev:
            econ["core_pce_mom_pct"] = round((pce_now - pce_prev) / pce_prev * 100, 2)
        nfp_now, nfp_prev, _ = _fred_latest_two(FRED_SERIES["nfp"])
        if nfp_now and nfp_prev:
            econ["nfp_change_k"] = round(nfp_now - nfp_prev, 0)
        unemp_now, unemp_prev, _ = _fred_latest_two(FRED_SERIES["unemployment"])
        if unemp_now is not None:
            econ["unemployment_rate"] = unemp_now
            econ["unemployment_chg"] = round(unemp_now - (unemp_prev or unemp_now), 2)
        fed_now, fed_prev, _ = _fred_latest_two(FRED_SERIES["fed_funds"])
        if fed_now is not None:
            econ["fed_funds_rate"] = fed_now
            econ["fed_funds_chg"] = round(fed_now - (fed_prev or fed_now), 2)
    except Exception as e:
        logger.warning(f"宏观经济数据处理出错: {e}")
    return econ


def get_macro_snapshot() -> dict:
    snap = get_macro_market_snapshot()
    snap.update(get_macro_economic_data())
    return snap


def vix_regime(vix) -> str:
    if vix is None: return "数据不可用"
    if vix < 13:  return "极度平静，警惕自满情绪"
    if vix < 17:  return "低恐慌，风险偏好健康"
    if vix < 22:  return "正常波动区间"
    if vix < 28:  return "波动加剧，市场定价不确定性"
    if vix < 35:  return "高度紧张，避险主导"
    return "极端恐慌，类似危机模式"


# ════════════════════════════════════════════════════════════════════════════
#  2. MFTSR 五维评分 — 精确对应 MFTSR_AI_Growth_Model_EN.xlsx
# ════════════════════════════════════════════════════════════════════════════

def score_macro(macro: dict) -> tuple:
    """宏观维度，20%权重。子指标：Fed Funds 18% | 10Y 18% | Core PCE 10% |
    Unemployment 10% | NFP 10% | ISM PMI 10% | DXY 5% | WTI 12% | WTI chg 7%。
    AI Policy / Geopolitics 已移除（仅作宏观简报文字背景，不计分）。"""
    sub_scores = []

    fed_rate = macro.get("fed_funds_rate")
    if fed_rate is not None:
        s = 90 if fed_rate <= 3 else 70 if fed_rate <= 4 else 45 if fed_rate <= 5 else 20
        sub_scores.append(("Fed Funds Rate", 0.18, s, f"{fed_rate:.2f}%"))
    else:
        sub_scores.append(("Fed Funds Rate", 0.18, 50, "数据缺失(需FRED API)"))

    us10y = macro.get("us10y")
    if us10y is not None:
        s = 90 if us10y <= 3.5 else 70 if us10y <= 4.5 else 45 if us10y <= 5 else 20
        sub_scores.append(("10Y Treasury Yield", 0.18, s, f"{us10y:.2f}%"))
    else:
        sub_scores.append(("10Y Treasury Yield", 0.18, 50, "数据缺失"))

    core_pce = macro.get("core_pce_mom_pct")
    if core_pce is not None:
        s = 100 if core_pce <= 2.5 else 75 if core_pce <= 3 else 45 if core_pce <= 4 else 20
        sub_scores.append(("Core PCE YoY", 0.10, s, f"{core_pce:+.2f}%"))
    else:
        sub_scores.append(("Core PCE YoY", 0.10, 50, "数据缺失(需FRED API)"))

    unemp = macro.get("unemployment_rate")
    if unemp is not None:
        s = 90 if 3.8 <= unemp <= 5 else 65 if unemp < 3.8 else 55 if unemp <= 6 else 25
        sub_scores.append(("Unemployment Rate", 0.10, s, f"{unemp:.1f}%"))
    else:
        sub_scores.append(("Unemployment Rate", 0.10, 50, "数据缺失"))

    nfp = macro.get("nfp_change_k")
    if nfp is not None:
        s = 90 if 100 <= nfp <= 250 else 55 if 0 <= nfp < 100 else 60 if nfp > 250 else 20
        sub_scores.append(("Nonfarm Payroll Change", 0.10, s, f"{nfp:+.0f}k"))
    else:
        sub_scores.append(("Nonfarm Payroll Change", 0.10, 50, "数据缺失"))

    sub_scores.append(("ISM Manufacturing PMI", 0.10, 50, "中性占位(无免费实时源，需手动更新)"))

    dxy = macro.get("dxy")
    if dxy is not None:
        s = 85 if dxy <= 102 else 65 if dxy <= 108 else 40
        sub_scores.append(("US Dollar Index (DXY)", 0.05, s, f"{dxy:.1f}"))
    else:
        sub_scores.append(("US Dollar Index (DXY)", 0.05, 50, "数据缺失"))

    crude = macro.get("crude")
    if crude is not None:
        s = 90 if 55 <= crude <= 80 else 60 if 80 < crude <= 95 else 30
        sub_scores.append(("WTI Oil Price", 0.12, s, f"${crude:.1f}"))
    else:
        sub_scores.append(("WTI Oil Price", 0.12, 50, "数据缺失"))

    crude_chg = macro.get("crude_chg_pct")
    if crude_chg is not None:
        s = 90 if abs(crude_chg) <= 5 else 60 if abs(crude_chg) <= 12 else 25
        sub_scores.append(("WTI 1M % Change", 0.07, s, f"{crude_chg:+.2f}%"))
    else:
        sub_scores.append(("WTI 1M % Change", 0.07, 50, "数据缺失"))

    weighted_total = sum(w * s for _, w, s, _ in sub_scores)
    return round(weighted_total), sub_scores


def score_fundamental(md: dict) -> tuple:
    """基本面维度，35%权重。Growth(40%): RevCAGR 20%/ProfitCAGR 20%.
    Earnings quality(35%): FCFMargin 12.25%/GrossMargin 10.5%/EPSGrowth 12.25%.
    Valuation(25%): ForwardPE 7.5%/PEG 10%/FCFYield 7.5%."""
    sub_scores = []

    rev_cagr = md.get("rev_cagr_proxy", 0)
    s = 100 if rev_cagr >= 30 else 85 if rev_cagr >= 20 else 65 if rev_cagr >= 10 else 30
    sub_scores.append(("Revenue Growth (YoY代理3Y CAGR)", 0.20, s, f"{rev_cagr:+.1f}%"))

    profit_cagr = md.get("profit_cagr_proxy", 0)
    s = 100 if profit_cagr >= 30 else 85 if profit_cagr >= 20 else 65 if profit_cagr >= 10 else 30
    sub_scores.append(("Profit Growth (YoY代理3Y CAGR)", 0.20, s, f"{profit_cagr:+.1f}%"))

    fcf_margin = md.get("fcf_margin", 0)
    s = 100 if fcf_margin >= 25 else 80 if fcf_margin >= 15 else 55 if fcf_margin >= 5 else 25
    sub_scores.append(("FCF Margin", 0.1225, s, f"{fcf_margin:.1f}%"))

    gross_margin = md.get("gross_margin", 0)
    s = 100 if gross_margin >= 70 else 80 if gross_margin >= 50 else 55 if gross_margin >= 30 else 25
    sub_scores.append(("Gross Margin", 0.105, s, f"{gross_margin:.1f}%"))

    eps_growth = md.get("eps_growth", 0)
    s = 100 if eps_growth >= 30 else 80 if eps_growth >= 15 else 55 if eps_growth >= 5 else 25
    sub_scores.append(("EPS Growth", 0.1225, s, f"{eps_growth:+.1f}%"))

    fwd_pe = md.get("forward_pe")
    if isinstance(fwd_pe, (int, float)) and fwd_pe > 0:
        s = 95 if fwd_pe <= 20 else 75 if fwd_pe <= 30 else 55 if fwd_pe <= 45 else 25
        sub_scores.append(("Forward PE", 0.075, s, f"{fwd_pe:.1f}x"))
    else:
        sub_scores.append(("Forward PE", 0.075, 50, "N/A"))

    peg = md.get("peg_ratio")
    if isinstance(peg, (int, float)) and peg > 0:
        s = 100 if peg <= 1 else 75 if peg <= 1.5 else 50 if peg <= 2.5 else 20
        sub_scores.append(("PEG Ratio", 0.10, s, f"{peg:.2f}"))
    else:
        sub_scores.append(("PEG Ratio", 0.10, 50, "N/A"))

    fcf_yield = md.get("fcf_yield", 0)
    s = 100 if fcf_yield >= 5 else 75 if fcf_yield >= 3 else 50 if fcf_yield >= 1.5 else 20
    sub_scores.append(("FCF Yield", 0.075, s, f"{fcf_yield:.1f}%"))

    weighted_total = sum(w * s for _, w, s, _ in sub_scores)
    return round(weighted_total), sub_scores


def score_technical(md: dict) -> tuple:
    """技术面维度，20%权重。Trend Composite 25% | 3M RelStrength vs QQQ 25%
    | RSI 15% | Volume Ratio 15% | 30D Vol 10% | Beta 10%."""
    if md.get("thin_history"):
        days = md.get("trading_days", 0)
        return 50, [("数据不足", 1.0, 50, f"上市仅{days}个交易日，技术面暂不可靠，中性处理")]

    sub_scores = []

    above_ma20, above_ma50, above_ma200 = md.get("above_ma20"), md.get("above_ma50"), md.get("above_ma200")
    price_vs_ma20 = 80 if above_ma20 else 30 if above_ma20 is not None else 50
    ma20_vs_ma50 = 75 if (above_ma20 and above_ma50) else 35 if (above_ma20 is not None and above_ma50 is not None) else 50
    ma50_vs_ma200 = 70 if (above_ma50 and above_ma200) else 40 if (above_ma50 is not None and above_ma200 is not None) else 50
    ma_cross = md.get("ma_cross")
    cross_score = 100 if ma_cross == "golden" else 20 if ma_cross == "death" else 60
    slope_confirm = 70 if (above_ma20 and above_ma50) else 40
    trend_composite = round(
        price_vs_ma20 * 0.25 + ma20_vs_ma50 * 0.20 + ma50_vs_ma200 * 0.20 +
        cross_score * 0.20 + slope_confirm * 0.15
    )
    cross_note = "金叉" if ma_cross == "golden" else "死叉" if ma_cross == "death" else "无交叉"
    sub_scores.append(("Trend Composite", 0.25, trend_composite,
                       f"MA5/20{cross_note}, 站上MA20:{above_ma20}, MA50:{above_ma50}, MA200:{above_ma200}"))

    rel_strength = md.get("rel_strength_3m")
    if isinstance(rel_strength, (int, float)):
        s = 100 if rel_strength >= 20 else 80 if rel_strength >= 10 else 60 if rel_strength >= 0 else 30
        sub_scores.append(("3M Relative Strength vs QQQ", 0.25, s, f"{rel_strength:+.1f}pp"))
    else:
        sub_scores.append(("3M Relative Strength vs QQQ", 0.25, 50, "N/A"))

    rsi = md.get("rsi")
    if isinstance(rsi, (int, float)):
        s = 100 if 55 <= rsi <= 70 else 75 if 45 <= rsi < 55 else 55 if 70 < rsi <= 80 else 35
        sub_scores.append(("RSI(14)", 0.15, s, f"{rsi:.0f}"))
    else:
        sub_scores.append(("RSI(14)", 0.15, 50, "N/A"))

    vol_ratio = md.get("vol_ratio", 1)
    s = 90 if 1.2 <= vol_ratio <= 2.5 else 70 if 1 <= vol_ratio < 1.2 else 50 if 0.7 <= vol_ratio < 1 else 30
    sub_scores.append(("Volume Ratio", 0.15, s, f"{vol_ratio:.2f}x"))

    vol_30d = md.get("vol_30d_annualized")
    if isinstance(vol_30d, (int, float)):
        s = 90 if vol_30d <= 25 else 65 if vol_30d <= 40 else 35
        sub_scores.append(("30D Annualised Volatility", 0.10, s, f"{vol_30d:.1f}%"))
    else:
        sub_scores.append(("30D Annualised Volatility", 0.10, 50, "N/A"))

    beta = md.get("beta", 1.0)
    s = 90 if beta <= 1.2 else 65 if beta <= 1.7 else 35
    sub_scores.append(("Beta vs QQQ/S&P", 0.10, s, f"{beta:.2f}"))

    weighted_total = sum(w * s for _, w, s, _ in sub_scores)
    return round(weighted_total), sub_scores


def score_sentiment(md: dict, macro: dict) -> tuple:
    """情绪维度，10%权重。VIX 60% | Put/Call 20% | Positive Analyst Rating 20%
    （Put/Call无免费实时源，用分析师评级方向近似代理，已标注）。"""
    sub_scores = []

    vix = macro.get("vix")
    if vix is not None:
        s = 80 if vix <= 15 else 90 if vix <= 20 else 65 if vix <= 25 else 35 if vix <= 35 else 15
        sub_scores.append(("VIX", 0.60, s, f"{vix:.1f}"))
    else:
        sub_scores.append(("VIX", 0.60, 50, "数据缺失"))

    analyst_rec = md.get("analyst_rec", 3)
    proxy_score = 80 if analyst_rec <= 1.8 else 65 if analyst_rec <= 2.3 else 45 if analyst_rec <= 3 else 30
    sub_scores.append(("Equity Put/Call Ratio (代理:分析师评级方向)", 0.20, proxy_score,
                       f"分析师评级{analyst_rec:.1f}(1=买入,5=卖出)，非真实Put/Call数据"))

    num_analysts = md.get("num_analysts", 0)
    if num_analysts > 0:
        positive_pct_proxy = max(0, min(100, (3.5 - analyst_rec) / 2.5 * 100))
        s = 90 if positive_pct_proxy >= 80 else 70 if positive_pct_proxy >= 65 else 50 if positive_pct_proxy >= 50 else 25
        sub_scores.append(("Positive Analyst Rating", 0.20, s, f"约{positive_pct_proxy:.0f}%(由评级均值估算)"))
    else:
        sub_scores.append(("Positive Analyst Rating", 0.20, 50, "无分析师覆盖数据"))

    weighted_total = sum(w * s for _, w, s, _ in sub_scores)
    return round(weighted_total), sub_scores


def score_risk(md: dict, macro: dict) -> tuple:
    """风险维度，15%权重。VIX 45% | 10Y 35% | Credit Spread 20%
    （Credit Spread需FRED的BAMLH0A0HYM2，未配置时用Beta+市值近似代理）。"""
    sub_scores = []

    vix = macro.get("vix")
    if vix is not None:
        s = 80 if vix <= 15 else 90 if vix <= 20 else 65 if vix <= 25 else 35 if vix <= 35 else 15
        sub_scores.append(("VIX", 0.45, s, f"{vix:.1f}"))
    else:
        sub_scores.append(("VIX", 0.45, 50, "数据缺失"))

    us10y = macro.get("us10y")
    if us10y is not None:
        s = 90 if us10y <= 3.5 else 70 if us10y <= 4.5 else 45 if us10y <= 5 else 20
        sub_scores.append(("10Y Treasury Yield", 0.35, s, f"{us10y:.2f}%"))
    else:
        sub_scores.append(("10Y Treasury Yield", 0.35, 50, "数据缺失"))

    beta = md.get("beta", 1.0)
    market_cap_b = md.get("market_cap_b", 0)
    if beta < 1.2 and market_cap_b > 100:
        proxy_score = 80
    elif beta < 1.7:
        proxy_score = 60
    else:
        proxy_score = 35
    sub_scores.append(("High Yield Credit Spread (代理:Beta+市值)", 0.20, proxy_score,
                       f"Beta={beta:.2f}, 市值${market_cap_b:.0f}B，非真实信用利差数据"))

    weighted_total = sum(w * s for _, w, s, _ in sub_scores)
    return round(weighted_total), sub_scores


def compute_mftsr(md: dict, macro: dict) -> dict:
    m, m_sub = score_macro(macro)
    f, f_sub = score_fundamental(md)
    t, t_sub = score_technical(md)
    s, s_sub = score_sentiment(md, macro)
    r, r_sub = score_risk(md, macro)

    composite = round(m * WEIGHTS["macro"] + f * WEIGHTS["fundamental"] +
                       t * WEIGHTS["technical"] + s * WEIGHTS["sentiment"] +
                       r * WEIGHTS["risk"])

    label, action, emoji = score_band(composite)

    dims = {"macro": m, "fundamental": f, "technical": t, "sentiment": s, "risk": r}
    alerts = [k for k, v in dims.items() if v < ALERT_DIM]

    price = md.get("price", 0)
    stop_pct = 0.06 if md.get("beta", 1) > 1.5 else 0.04
    stop_loss = round(price * (1 - stop_pct), 2)
    edge = (composite - 50) / 50
    position_pct = max(0, min(20, round(edge * 15)))

    return {
        "composite": composite, "label": label, "action": action, "emoji": emoji,
        "macro": m, "fundamental": f, "technical": t, "sentiment": s, "risk": r,
        "macro_sub": m_sub, "fundamental_sub": f_sub, "technical_sub": t_sub,
        "sentiment_sub": s_sub, "risk_sub": r_sub,
        "alerts": alerts, "stop_loss": stop_loss, "position_pct": position_pct,
    }


# ════════════════════════════════════════════════════════════════════════════
#  3. Claude AI 文字解读 — 五个独立段落
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_MACRO = """你是MFTSR Alpha系统的宏观策略分析师，融合Ray Dalio的宏观债务周期思维。
生成今日宏观简报，覆盖经济数据(CPI/非农/PMI)、美联储政策动态、VIX市场情绪含义、
油价与美债收益率走势。地缘政治/关税风险作为定性背景提及（不带具体评分权重，
基于你的知识给出合理评估，并说明这是定性判断、非实时新闻）。
风格：直接、精准、专业；数字优先；不确定时说"信号混合"；篇幅紧凑；简体中文。"""

SYSTEM_PROMPT_STOCK = """你是MFTSR Alpha系统的个股分析师，融合Peter Lynch基本面哲学、
Stanley Druckenmiller趋势跟随、Jim Simons量化信号体系。
今日宏观背景已在另一份简报中说明。你需要为这只股票生成5个独立段落，分别对应
基本面/技术面/情绪/风险/宏观影响五个维度，每段明确给出该维度的评分（已提供，
直接引用），并结合具体数据展开2-3句分析。风格：直接、精准、专业，数字优先，
篇幅紧凑（这是多只股票中的一只），简体中文。"""


def build_macro_prompt(macro: dict) -> str:
    fred_note = "" if FRED_API_KEY else "\n（注：未配置FRED API，CPI/非农/失业率/Fed利率为缺失状态，宏观评分对应子项按中性50分处理，建议配置FRED_API_KEY以获得真实数据）"
    return f"""请生成今日美股收盘宏观简报。

═══ 市场数据 ═══
VIX: {macro.get('vix','N/A')} ({macro.get('vix_chg_pct',0):+.2f}%) — {vix_regime(macro.get('vix'))}
DXY: {macro.get('dxy','N/A')}
US10Y: {macro.get('us10y','N/A')}%
标普500: {macro.get('sp500_chg_pct',0):+.2f}%  纳指: {macro.get('nasdaq_chg_pct',0):+.2f}%
WTI原油: ${macro.get('crude','N/A')} ({macro.get('crude_chg_pct',0):+.2f}%)
黄金: {macro.get('gold','N/A')}
Fed Funds Rate: {macro.get('fed_funds_rate','N/A')}
CPI环比: {macro.get('cpi_mom_pct','N/A')}
非农变化: {macro.get('nfp_change_k','N/A')}{fred_note}

请按以下结构输出（不加多余章节），用简体中文：

【今日宏观核心判断】一句话总结
【经济数据与美联储】2-3点
【VIX与风险偏好】1-2点
【地缘政治与关税背景】2-3点（基于你的知识定性评估，明确标注这是定性判断非实时新闻，且此项不计入任何维度的数值评分）
【对美股整体影响】1-2点
【本周关键事件】1-2个需关注的事件"""


def get_macro_commentary(macro: dict) -> str:
    if not ANTHROPIC_KEY:
        return "[AI未配置]"
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        prompt = build_macro_prompt(macro)
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=900,
            system=SYSTEM_PROMPT_MACRO, messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.error(f"宏观简报生成失败: {e}")
        return f"[AI解读暂时不可用: {e}]"


def _fmt(val, decimals=1):
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def _format_sub_scores(sub_scores: list) -> str:
    lines = []
    for label, weight, score, detail in sub_scores:
        lines.append(f"  - {label} (权重{weight*100:.1f}%): 评分{score} | {detail}")
    return "\n".join(lines)


def build_stock_prompt(ticker: dict, md: dict, scores: dict, macro_headline: str) -> str:
    sym, name = ticker["sym"], ticker["name"]

    thin_note = ""
    if md.get("thin_history"):
        days = md.get("trading_days", 0)
        thin_note = f"\n⚠️ 注意：{sym}上市仅{days}个交易日，技术面数据不足，请在技术面段落明确说明"

    alerts_str = f"\n⚠️ 风险警报维度（评分<30）: {', '.join(scores['alerts'])}" if scores.get("alerts") else ""

    return f"""【个股分析】{sym} — {name}

═══ 价格 ═══
${md['price']} ({md['change_pct']:+.2f}%){thin_note}

═══ 基本面子指标明细（35%权重）═══
{_format_sub_scores(scores['fundamental_sub'])}
基本面综合评分: {scores['fundamental']}/100

═══ 技术面子指标明细（20%权重）═══
{_format_sub_scores(scores['technical_sub'])}
技术面综合评分: {scores['technical']}/100

═══ 情绪面子指标明细（10%权重）═══
{_format_sub_scores(scores['sentiment_sub'])}
情绪面综合评分: {scores['sentiment']}/100

═══ 风险面子指标明细（15%权重）═══
{_format_sub_scores(scores['risk_sub'])}
风险面综合评分: {scores['risk']}/100

═══ MFTSR综合 ═══
{scores['composite']}/100 → {scores['label']}（{scores['action']}）
宏观维度评分: {scores['macro']}/100（占20%权重，已计入综合分）{alerts_str}

═══ 今日宏观要点（供参考）═══
{macro_headline}

请生成5个独立段落，每段开头用【】标注维度名+评分，全部简体中文：

【基本面 - 评分{scores['fundamental']}/100】2-3句话，结合Forward PE/PEG/营收利润增速展开
【技术面 - 评分{scores['technical']}/100】2-3句话，结合趋势/金叉死叉/RSI/相对强弱展开
【情绪面 - 评分{scores['sentiment']}/100】2-3句话，结合VIX/分析师评级展开
【风险面 - 评分{scores['risk']}/100】2-3句话，结合Beta/波动率/具体风险点展开
【宏观影响 - 评分{scores['macro']}/100】1-2句话：今日宏观环境对这只股票的具体影响

【操作建议】
• 综合评分: {scores['composite']}/100 → {scores['action']}
• 止损参考: ${scores['stop_loss']}
• 仓位建议: 不超过{scores['position_pct']}%"""


def get_stock_commentary(ticker: dict, md: dict, scores: dict, macro_headline: str) -> str:
    if not ANTHROPIC_KEY:
        return _fallback_commentary(ticker, md, scores)
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        prompt = build_stock_prompt(ticker, md, scores, macro_headline)
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1000,
            system=SYSTEM_PROMPT_STOCK, messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.error(f"Claude API 调用失败 {ticker['sym']}: {e}")
        return _fallback_commentary(ticker, md, scores)


def _fallback_commentary(ticker: dict, md: dict, scores: dict) -> str:
    sym = ticker["sym"]
    return (f"[AI离线] {sym} 规则评分摘要\n"
            f"【基本面 - 评分{scores['fundamental']}/100】数据见上方明细\n"
            f"【技术面 - 评分{scores['technical']}/100】数据见上方明细\n"
            f"【情绪面 - 评分{scores['sentiment']}/100】数据见上方明细\n"
            f"【风险面 - 评分{scores['risk']}/100】数据见上方明细\n"
            f"【宏观影响 - 评分{scores['macro']}/100】见宏观简报\n"
            f"止损${scores['stop_loss']} | 仓位≤{scores['position_pct']}%")


# ════════════════════════════════════════════════════════════════════════════
#  4. Telegram 发送
# ════════════════════════════════════════════════════════════════════════════

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def _send(text: str) -> bool:
    if not SEND_TELEGRAM:
        logger.info("SEND_TELEGRAM=false，跳过Telegram发送（仅生成Dashboard数据）")
        return True
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.error("Telegram 未配置，跳过发送")
        return False
    try:
        resp = requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=15)
        data = resp.json()
        if not data.get("ok"):
            logger.error(f"Telegram 错误: {data}")
            return False
        return True
    except requests.RequestException as e:
        logger.error(f"Telegram 网络错误: {e}")
        return False


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _score_bar(score: int, width: int = 8) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def build_macro_message(macro: dict, commentary: str) -> str:
    uk_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    vix = macro.get("vix", "N/A")
    vix_chg = macro.get("vix_chg_pct", 0)
    vix_icon = "😱" if isinstance(vix, (int, float)) and vix > 25 else ("😰" if isinstance(vix, (int, float)) and vix > 18 else "😌")

    lines = [
        f"<b>◈ MFTSR ALPHA — 🌐 今日宏观简报</b>",
        f"🕐 {uk_now} (UK时间)",
        f"{'─'*32}",
        f"  {vix_icon} VIX: <b>{vix}</b> ({vix_chg:+.2f}%) — {vix_regime(vix if isinstance(vix, (int, float)) else None)}",
        f"  💵 DXY: <b>{macro.get('dxy','N/A')}</b>",
        f"  📈 S&amp;P500: <b>{macro.get('sp500_chg_pct',0):+.2f}%</b>  纳指: <b>{macro.get('nasdaq_chg_pct',0):+.2f}%</b>",
        f"  🏦 US10Y: <b>{macro.get('us10y','N/A')}%</b>",
        f"  🛢 WTI原油: <b>${macro.get('crude','N/A')}</b> ({macro.get('crude_chg_pct',0):+.2f}%)",
        f"  🥇 黄金: <b>{macro.get('gold','N/A')}</b>",
    ]
    if macro.get("fed_funds_rate") is not None:
        lines.append(f"  🏛 Fed Funds Rate: <b>{macro['fed_funds_rate']:.2f}%</b>")
    if macro.get("cpi_mom_pct") is not None:
        lines.append(f"  📊 CPI环比: <b>{macro['cpi_mom_pct']:+.2f}%</b>")

    lines.append(f"{'─'*32}")
    lines.append(_escape(commentary))
    lines.append(f"{'─'*32}")
    lines.append(SCORE_LEGEND_ZH)

    return "\n".join(lines)


def build_stock_message(ticker: dict, md: dict, scores: dict, commentary: str) -> str:
    sym, name = ticker["sym"], ticker["name"]
    chg_arrow = "▲" if md["change_pct"] >= 0 else "▼"

    alert_line = f"\n⚠️ <b>风险警报</b>: {', '.join(scores['alerts'])}维度评分极低" if scores.get("alerts") else ""

    return (
        f"\n{scores['emoji']} <b>{sym}</b>  {_escape(name)}\n"
        f"  💰 <b>${md['price']}</b>  {chg_arrow}{abs(md['change_pct']):.2f}%\n"
        f"\n<b>MFTSR综合: {scores['composite']}/100</b>  [{_score_bar(scores['composite'])}]  "
        f"<b>{scores['label']}</b>\n"
        f"  宏观{scores['macro']:>3}(20%)  基本面{scores['fundamental']:>3}(35%)\n"
        f"  技术面{scores['technical']:>3}(20%)  情绪{scores['sentiment']:>3}(10%)  风险{scores['risk']:>3}(15%)"
        f"{alert_line}\n"
        f"\n{'─'*28}\n{_escape(commentary)}\n{'─'*28}\n"
        f"🎯 <b>{scores['action']}</b>  |  止损 ${scores['stop_loss']}  |  仓位 ≤{scores['position_pct']}%\n"
    )


def build_footer(results: list) -> str:
    header = "\n📋 <b>本次报告汇总</b>\n" + "─"*28 + "\n"
    rows = []
    for r in sorted(results, key=lambda x: x["scores"]["composite"], reverse=True):
        emoji = r["scores"]["emoji"]
        rows.append(
            f"  {emoji} <b>{r['ticker']['sym']:<8}</b>  {r['scores']['composite']:>3}/100  "
            f"{r['scores']['label']}  {r['md']['change_pct']:+.2f}%"
        )
    disclaimer = "\n❓ 以上内容仅供参考，不构成投资建议。部分子指标（Put/Call、信用利差、PMI）因无免费实时数据源使用代理近似，已在AI解读中标注。"
    return header + "\n".join(rows) + "\n" + "─"*28 + disclaimer


# ════════════════════════════════════════════════════════════════════════════
#  5.5. 独立监测模块 — Storage / AI Capex / Semiconductor / China Policy /
#       Energy Transition（完全独立于MFTSR评分，权重精确对应Excel模型）
#
#  这5个模块的子指标全部是定性判断（DRAM价格趋势、HBM市场动态、超大规模厂商
#  资本支出增速等），没有任何免费实时API能直接给出0-100的数字。这里用Claude
#  基于其知识生成定性评分，每个分数都在UI和数据里明确标注"AI定性判断，非实时
#  数据"，并建议用户结合最新行业资讯复核。
# ════════════════════════════════════════════════════════════════════════════

MONITOR_DEFS = {
    "storage": {
        "label": "Storage Cycle Monitor", "label_zh": "存储周期监测",
        "applicable": ["MU"],
        "applicable_note": "MU, SK Hynix, Samsung Memory, Western Digital, SanDisk（仅MU在当前股票池内）",
        "caveat": "专为存储芯片(DRAM/NAND/HBM)相关投资设计，不适用于软件/互联网等非存储敞口标的。",
        "metrics": [
            {"key": "dram_trend", "label": "DRAM Price Trend", "weight": 0.30,
             "prompt": "DRAM现货/合约价格近期趋势（涨价/跌价/稳定，参考TrendForce/DRAMeXchange类公开报道方向）"},
            {"key": "nand_trend", "label": "NAND Price Trend", "weight": 0.20,
             "prompt": "NAND闪存现货/合约价格近期趋势"},
            {"key": "hbm_trend", "label": "HBM Market Trend", "weight": 0.25,
             "prompt": "HBM(高带宽存储)市场供需、定价、客户需求趋势"},
            {"key": "inventory_cycle", "label": "Inventory Cycle", "weight": 0.15,
             "prompt": "存储行业渠道库存周期（紧张/正常化/累库/严重过剩）"},
            {"key": "supply_discipline", "label": "Supply Discipline", "weight": 0.10,
             "prompt": "主要存储厂商的产能扩张纪律性（克制扩产/激进扩产）"},
        ],
    },
    "ai_capex": {
        "label": "AI Capex Monitor", "label_zh": "AI资本支出监测",
        "applicable": ["NVDA", "AVGO", "MRVL", "MU"],
        "applicable_note": "NVDA, AVGO, MRVL, MU（+ AMD/VRT/ANET作为参考，未在当前股票池）",
        "caveat": "追踪超大规模云厂商AI基础设施支出周期，适用于AI算力/网络/存储硬件敞口标的。",
        "metrics": [
            {"key": "hyperscaler_capex", "label": "Hyperscaler Capex Growth", "weight": 0.30,
             "prompt": "微软/谷歌/亚马逊/Meta等超大规模云厂商资本支出增速指引趋势"},
            {"key": "ai_server_demand", "label": "AI Server Demand", "weight": 0.20,
             "prompt": "AI服务器订单积压、交付周期长短反映的需求强弱"},
            {"key": "gpu_demand", "label": "GPU Demand", "weight": 0.20,
             "prompt": "GPU分配紧张程度、定价能力"},
            {"key": "cloud_ai_spending", "label": "Cloud AI Spending", "weight": 0.15,
             "prompt": "云厂商AI相关收入增长轨迹"},
            {"key": "enterprise_ai_adoption", "label": "Enterprise AI Adoption", "weight": 0.15,
             "prompt": "企业AI部署/预算投入节奏"},
        ],
    },
    "semiconductor": {
        "label": "Semiconductor Cycle Monitor", "label_zh": "半导体周期监测",
        "applicable": ["NVDA", "AVGO", "MRVL"],
        "applicable_note": "NVDA, AVGO, MRVL（+ AMD/TSM作为参考，未在当前股票池）",
        "caveat": "追踪半导体产能/需求周期，独立于单一公司基本面。",
        "metrics": [
            {"key": "foundry_utilization", "label": "Foundry Utilization", "weight": 0.25,
             "prompt": "台积电/三星等领先晶圆代工厂产能利用率水平"},
            {"key": "inventory_cycle_semi", "label": "Inventory Cycle", "weight": 0.20,
             "prompt": "半导体产业链渠道库存周期"},
            {"key": "ai_chip_demand", "label": "AI Chip Demand", "weight": 0.25,
             "prompt": "AI芯片订单可见度、分配紧张程度"},
            {"key": "data_center_growth", "label": "Data Center Growth", "weight": 0.20,
             "prompt": "数据中心新建/扩建节奏"},
            {"key": "lead_time", "label": "Lead Time", "weight": 0.10,
             "prompt": "芯片供应链交付周期长短"},
        ],
    },
    "china_policy": {
        "label": "China Policy Monitor", "label_zh": "中国政策监测",
        "applicable": [],  # 美股清单中无A股标的，此monitor主要服务A股报告
        "applicable_note": "适用于所有A股持仓及中国敞口标的（详见A股早报）",
        "caveat": "追踪中国宏观政策背景，非个股基本面，主要供A股投资参考。",
        "metrics": [
            {"key": "fiscal_stimulus", "label": "Fiscal Stimulus", "weight": 0.25,
             "prompt": "中国财政刺激力度与落地节奏"},
            {"key": "monetary_policy", "label": "Monetary Policy", "weight": 0.25,
             "prompt": "央行货币政策取向（宽松/收紧，LPR/存准率方向）"},
            {"key": "property_policy", "label": "Property Policy", "weight": 0.20,
             "prompt": "房地产支持政策力度"},
            {"key": "industrial_policy", "label": "Industrial Policy", "weight": 0.20,
             "prompt": "半导体/新能源等战略行业产业政策支持力度"},
            {"key": "us_china_relations", "label": "US-China Relations", "weight": 0.10,
             "prompt": "中美关税/出口管制关系走向"},
        ],
    },
    "energy_transition": {
        "label": "Energy Transition Monitor", "label_zh": "能源转型监测",
        "applicable": ["NEE", "GEV", "CEG"],
        "applicable_note": "NEE, GEV, CEG（+ 皖能电力作为A股参考）",
        "caveat": "追踪电力需求与电网投资周期，适用于电力/电网设备相关持仓。",
        "metrics": [
            {"key": "power_demand_growth", "label": "Power Demand Growth", "weight": 0.25,
             "prompt": "整体电网负荷增长轨迹"},
            {"key": "data_center_elec_demand", "label": "Data Center Electricity Demand", "weight": 0.25,
             "prompt": "数据中心电力消耗增长预测"},
            {"key": "grid_investment", "label": "Grid Investment", "weight": 0.25,
             "prompt": "输配电网基础设施资本支出趋势"},
            {"key": "energy_prices", "label": "Energy Prices", "weight": 0.25,
             "prompt": "批发电价走势"},
        ],
    },
}

MONITOR_BANDS = [
    (80, 101, "Strong Upcycle", "强劲上行周期"),
    (65, 80,  "Healthy Recovery", "健康复苏"),
    (50, 65,  "Neutral", "中性"),
    (35, 50,  "Weakening", "走弱"),
    (0,  35,  "Downcycle", "下行周期"),
]


def monitor_band(score: int) -> tuple:
    for lo, hi, label_en, label_zh in MONITOR_BANDS:
        if lo <= score < hi:
            return label_en, label_zh
    return MONITOR_BANDS[-1][2], MONITOR_BANDS[-1][3]


SYSTEM_PROMPT_MONITOR = """你是行业周期监测分析师。任务：针对给定的具体指标，基于你的知识给出
一个0-100的定性评分，并用1句话说明理由。这些指标本质上没有实时数字数据源，你的评分是
基于行业认知的合理估计，不是精确数据。请明确、直接给出评分，不要说"无法判断"——
即使信息不完整，也要基于已知趋势给出一个合理的中性偏向判断（如果真的没有把握，用50分
表示中性，但仍需给出理由）。

输出格式严格为：
评分: <0-100的整数>
理由: <一句话，不超过40字，简体中文>"""


def get_monitor_metric_score(metric_prompt: str) -> tuple:
    """调用Claude给单个监测指标打分。返回 (score, rationale)。"""
    if not ANTHROPIC_KEY:
        return 50, "AI未配置，中性占位"
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=150,
            system=SYSTEM_PROMPT_MONITOR,
            messages=[{"role": "user", "content": f"请评估：{metric_prompt}"}],
        )
        text = msg.content[0].text
        score_match = None
        rationale = text.strip()
        for line in text.split("\n"):
            if line.strip().startswith("评分"):
                digits = "".join(c for c in line if c.isdigit())
                if digits:
                    score_match = max(0, min(100, int(digits)))
            elif line.strip().startswith("理由"):
                rationale = line.split("理由:", 1)[-1].split("理由：", 1)[-1].strip()
        return (score_match if score_match is not None else 50), rationale
    except Exception as e:
        logger.warning(f"监测指标打分失败: {e}")
        return 50, f"AI调用失败: {e}"


def compute_monitor_scores() -> dict:
    """
    为5个独立监测模块逐项打分。每个子指标单独调用一次Claude（共17次调用，
    跨5个模块），全部是定性AI估计，非实时数据，已在每条结果里标注。
    """
    results = {}
    for key, mon in MONITOR_DEFS.items():
        logger.info(f"  监测模块: {mon['label']}...")
        sub_results = []
        weighted_total = 0
        for m in mon["metrics"]:
            score, rationale = get_monitor_metric_score(m["prompt"])
            sub_results.append({
                "key": m["key"], "label": m["label"], "weight": m["weight"],
                "score": score, "rationale": rationale,
            })
            weighted_total += score * m["weight"]
            time.sleep(1.5)
        composite = round(weighted_total)
        band_en, band_zh = monitor_band(composite)
        results[key] = {
            "label": mon["label"], "label_zh": mon["label_zh"],
            "applicable": mon["applicable"], "applicable_note": mon["applicable_note"],
            "caveat": mon["caveat"],
            "composite": composite, "band": band_en, "band_zh": band_zh,
            "metrics": sub_results,
            "is_ai_estimate": True,
        }
        logger.info(f"    {mon['label']}: {composite}/100 ({band_en})")
    return results


# ════════════════════════════════════════════════════════════════════════════
#  6. Dashboard JSON 导出
# ════════════════════════════════════════════════════════════════════════════

def export_dashboard_json(macro, macro_commentary, results, monitors=None, path="docs/dashboard_data.json"):
    def safe(v):
        if v is None:
            return None
        if isinstance(v, (int, float, str, bool)):
            return v
        return str(v)

    def serialize_sub_scores(sub_scores):
        return [{"label": label, "weight": weight, "score": score, "detail": detail}
                for label, weight, score, detail in sub_scores]

    payload = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at_uk": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "weights": WEIGHTS,
        "macro": {
            "commentary": macro_commentary,
            "vix": safe(macro.get("vix")),
            "vix_chg_pct": safe(macro.get("vix_chg_pct")),
            "dxy": safe(macro.get("dxy")),
            "us10y": safe(macro.get("us10y")),
            "sp500_chg_pct": safe(macro.get("sp500_chg_pct")),
            "nasdaq_chg_pct": safe(macro.get("nasdaq_chg_pct")),
            "crude": safe(macro.get("crude")),
            "crude_chg_pct": safe(macro.get("crude_chg_pct")),
            "gold": safe(macro.get("gold")),
            "fed_funds_rate": safe(macro.get("fed_funds_rate")),
            "cpi_mom_pct": safe(macro.get("cpi_mom_pct")),
            "nfp_change_k": safe(macro.get("nfp_change_k")),
        },
        "monitors": monitors or {},
        "stocks": [],
    }

    for r in results:
        md, scores, ticker = r["md"], r["scores"], r["ticker"]
        payload["stocks"].append({
            "sym": ticker["sym"],
            "name": ticker["name"],
            "sector": ticker.get("sector", ""),
            "price": safe(md.get("price")),
            "change_pct": safe(md.get("change_pct")),
            "rsi": safe(md.get("rsi")),
            "pe_ratio": safe(md.get("pe_ratio")),
            "forward_pe": safe(md.get("forward_pe")),
            "peg_ratio": safe(md.get("peg_ratio")),
            "ma5": safe(md.get("ma5")), "ma20": safe(md.get("ma20")), "ma50": safe(md.get("ma50")),
            "ma_cross": safe(md.get("ma_cross")),
            "vol_ratio": safe(md.get("vol_ratio")),
            "rel_strength_3m": safe(md.get("rel_strength_3m")),
            "vol_30d_annualized": safe(md.get("vol_30d_annualized")),
            "target_price": safe(md.get("target_price")),
            "target_high": safe(md.get("target_high")),
            "target_low": safe(md.get("target_low")),
            "num_analysts": safe(md.get("num_analysts")),
            "upside": safe(md.get("upside")),
            "rev_growth": safe(md.get("rev_growth")),
            "eps_growth": safe(md.get("eps_growth")),
            "gross_margin": safe(md.get("gross_margin")),
            "fcf_yield": safe(md.get("fcf_yield")),
            "fcf_margin": safe(md.get("fcf_margin")),
            "inst_pct": safe(md.get("inst_pct")),
            "short_ratio": safe(md.get("short_ratio")),
            "analyst_rec": safe(md.get("analyst_rec")),
            "beta": safe(md.get("beta")),
            "market_cap_b": safe(md.get("market_cap_b")),
            "thin_history": safe(md.get("thin_history")),
            "trading_days": safe(md.get("trading_days")),
            "scores": {
                "composite": scores["composite"],
                "label": scores["label"],
                "action": scores["action"],
                "macro": scores["macro"],
                "fundamental": scores["fundamental"],
                "technical": scores["technical"],
                "sentiment": scores["sentiment"],
                "risk": scores["risk"],
                "stop_loss": scores["stop_loss"],
                "position_pct": scores["position_pct"],
                "alerts": scores.get("alerts", []),
                "fundamental_sub": serialize_sub_scores(scores["fundamental_sub"]),
                "technical_sub": serialize_sub_scores(scores["technical_sub"]),
                "sentiment_sub": serialize_sub_scores(scores["sentiment_sub"]),
                "risk_sub": serialize_sub_scores(scores["risk_sub"]),
                "macro_sub": serialize_sub_scores(scores["macro_sub"]),
            },
            "commentary": r["commentary"],
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Dashboard JSON导出完成: {path} ({len(payload['stocks'])}只股票)")


# ════════════════════════════════════════════════════════════════════════════
#  7. 主流程
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("开始生成美股收盘报告（v5，精确对齐Excel框架）...")
    logger.info(f"股票池: {[t['sym'] for t in WATCHLIST]}")
    logger.info(f"权重: {WEIGHTS}")

    macro = get_macro_snapshot()
    logger.info(f"VIX={macro.get('vix')} DXY={macro.get('dxy')} US10Y={macro.get('us10y')}")

    macro_commentary = get_macro_commentary(macro)
    macro_headline = macro_commentary.split("\n")[0] if macro_commentary else "宏观环境数据见上方简报"
    time.sleep(1)

    results = []
    failures = []
    for ticker in WATCHLIST:
        sym = ticker["sym"]
        logger.info(f"  分析 {sym}...")
        try:
            md = get_market_data(sym)
            scores = compute_mftsr(md, macro)
            commentary = get_stock_commentary(ticker, md, scores, macro_headline)
            results.append({"ticker": ticker, "md": md, "scores": scores, "commentary": commentary})
            logger.info(f"  {sym}: composite={scores['composite']} ({scores['label']})")
        except Exception as e:
            logger.error(f"  {sym} 失败: {e}")
            failures.append(f"{sym}: {e}")
        time.sleep(2)

    if not results:
        logger.error(f"没有生成任何结果，终止发送。失败详情（共{len(failures)}只）：")
        for f in failures:
            logger.error(f"  - {f}")
        # 即使Telegram什么都没收到，也尝试发一条简短警报，让问题至少能被看到
        try:
            _send(
                f"🚨 <b>美股收盘报告运行失败</b>\n"
                f"全部{len(WATCHLIST)}只股票均分析失败，未生成任何报告。\n"
                f"首个错误: {_escape(failures[0]) if failures else '未知'}\n"
                f"请检查GitHub Actions日志排查具体原因。"
            )
        except Exception:
            pass
        sys.exit(1)

    header_sent = _send(build_macro_message(macro, macro_commentary))
    if not header_sent:
        logger.error(
            "首条消息发送失败，可能是 TELEGRAM_TOKEN 或 CHAT_ID 配置错误。"
            "终止本次报告发送，workflow将标记为失败以便排查。"
        )
        sys.exit(1)
    time.sleep(1.5)

    results.sort(key=lambda r: r["scores"]["composite"], reverse=True)
    for r in results:
        msg = build_stock_message(r["ticker"], r["md"], r["scores"], r["commentary"])
        if len(msg) > 4000:
            for i in range(0, len(msg), 3900):
                _send(msg[i:i+3900]); time.sleep(0.5)
        else:
            _send(msg)
        time.sleep(1.5)

    _send(build_footer(results))
    logger.info(f"报告发送完成：宏观简报 + {len(results)}只个股 + 汇总 ✅")

    monitors = {}
    try:
        logger.info("开始计算5个独立监测模块（Storage/AI Capex/Semiconductor/China Policy/Energy Transition）...")
        monitors = compute_monitor_scores()
    except Exception as e:
        logger.error(f"监测模块计算失败（不影响主报告，Dashboard将不显示Monitor板块): {e}")

    try:
        export_dashboard_json(macro, macro_commentary, results, monitors=monitors)
    except Exception as e:
        logger.error(f"Dashboard JSON导出失败（不影响Telegram报告已发送）: {e}")


if __name__ == "__main__":
    main()
