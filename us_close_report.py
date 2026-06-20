"""
market_report.py — MFTSR Alpha · GitHub Actions 版本 / Cloud Edition

无需本地环境，由 GitHub Actions 在云端定时运行。
融合五维评分模型：宏观(Macro) / 基本面(Fundamental) / 技术面(Technical) /
情绪(Sentiment) / 风险(Risk)，规则打分 + Claude AI 文字解读，结果推送到 Telegram。
每次运行发送两条独立消息：一条纯中文，一条纯英文。

宏观维度覆盖（Macro dimension covers）：
  经济数据 Economic data    — CPI / NFP (非农) / PMI
  美联储动态 Fed policy     — Fed Funds Rate, FOMC stance
  地缘政治 Geopolitics      — Iran/Middle East tension, tariffs (via AI commentary)
  VIX 水平及含义 VIX regime — explicit interpretation layer, not just the number
  油价/国债收益率 Oil & Yields — WTI crude, US10Y, yield curve

环境变量（在 GitHub Secrets 中配置）：
    ANTHROPIC_KEY   — Claude API key
    TELEGRAM_TOKEN  — Telegram bot token
    CHAT_ID         — Telegram chat id（群组为负数）
    FRED_API_KEY    — (可选) FRED API key，用于 CPI/NFP/PMI/Fed Funds Rate 数据
                       免费申请：https://fred.stlouisfed.org/docs/api/api_key.html
                       若未提供，宏观经济数据部分会跳过，仅用市场化指标(VIX/DXY/油价等)

可选环境变量（在 workflow yml 中通过 env 传入，无需写入 Secrets）：
    REPORT_TYPE     — premarket | midday | close
    WATCHLIST       — 逗号分隔的股票代码
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone

import requests
import numpy as np
import yfinance as yf
import anthropic

from shared_config import WEIGHTS, SCORE_BUY, SCORE_HOLD, ALERT_DIM, US_WATCHLIST as DEFAULT_WATCHLIST

# ─── 日志 ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mftsr")

# ─── 环境变量 / 配置 ──────────────────────────────────────────────────────────
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
FRED_API_KEY   = os.environ.get("FRED_API_KEY", "")
REPORT_TYPE    = os.environ.get("REPORT_TYPE", "close")   # premarket | midday | close
MAX_TICKERS    = int(os.environ.get("MAX_TICKERS", "15")) # 现在每天只发一次，默认展示全部15只

# ── 股票池：从 shared_config.py 引入，可通过环境变量 WATCHLIST 覆盖 ──────────

_watchlist_env = os.environ.get("WATCHLIST", "")
if _watchlist_env.strip():
    WATCHLIST = [{"sym": s.strip().upper(), "name": s.strip().upper(), "sector": "Custom"}
                 for s in _watchlist_env.split(",") if s.strip()]
else:
    WATCHLIST = DEFAULT_WATCHLIST

# MFTSR 五维权重 — 从 shared_config.py 引入（WEIGHTS, SCORE_BUY, SCORE_HOLD, ALERT_DIM）

REPORT_LABELS_ZH = {
    "premarket": "🌅 盘前分析 PRE-MARKET",
    "midday":    "☀️ 盘中快报 MIDDAY PULSE",
    "close":     "🌆 尾盘总结 CLOSE SUMMARY",
}
REPORT_LABELS_EN = {
    "premarket": "🌅 Pre-Market Briefing",
    "midday":    "☀️ Midday Pulse",
    "close":     "🌆 Close Summary",
}


# ════════════════════════════════════════════════════════════════════════════
#  1. 市场数据抓取 — Market Data
# ════════════════════════════════════════════════════════════════════════════

def get_market_data(sym: str) -> dict:
    """抓取单只股票的价格、技术指标、基本面数据。
    对新上市（价格历史<200天）的股票，长周期指标(MA200等)会标记为'N/A'而非用不足数据硬算。
    """
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="90d", interval="1d")
        if hist.empty:
            return _fallback(sym)

        close, volume = hist["Close"], hist["Volume"]
        n_days = len(close)
        latest = float(close.iloc[-1])
        prev   = float(close.iloc[-2]) if len(close) > 1 else latest
        change_pct = (latest - prev) / prev * 100

        hi_52w = float(close.tail(252).max())
        lo_52w = float(close.tail(252).min())

        # ── 长周期均线：数据不足时显式标记，而非用过短窗口硬算出误导性数字 ──
        ma20  = float(close.rolling(20).mean().iloc[-1])  if n_days >= 20  else None
        ma50  = float(close.rolling(50).mean().iloc[-1])  if n_days >= 50  else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if n_days >= 200 else None
        thin_history = n_days < 50   # 上市不足50个交易日，技术面评分需调整

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float(100 - (100 / (1 + rs)).iloc[-1]) if n_days >= 14 and not rs.isna().iloc[-1] else None

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = float((macd_line - signal_line).iloc[-1])

        avg_vol   = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
        vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_up  = float((bb_mid + 2 * bb_std).iloc[-1])
        bb_low = float((bb_mid - 2 * bb_std).iloc[-1])
        bb_pct = (latest - bb_low) / (bb_up - bb_low) * 100 if (bb_up - bb_low) > 0 else 50.0

        info = ticker.info
        pe_ratio   = info.get("trailingPE") or info.get("forwardPE") or 0
        rev_growth = (info.get("revenueGrowth") or 0) * 100
        eps_growth = (info.get("earningsGrowth") or 0) * 100
        fcf_yield  = 0
        if info.get("freeCashflow") and info.get("marketCap"):
            fcf_yield = info["freeCashflow"] / info["marketCap"] * 100
        short_ratio  = info.get("shortRatio") or 0
        inst_pct     = (info.get("heldPercentInstitutions") or 0) * 100
        market_cap_b = (info.get("marketCap") or 0) / 1e9
        beta         = info.get("beta") or 1.0
        analyst_rec  = info.get("recommendationMean") or 3.0
        target_price = info.get("targetMeanPrice") or latest
        upside       = (target_price - latest) / latest * 100 if latest > 0 else 0

        return {
            "sym": sym, "price": round(latest, 2), "change_pct": round(change_pct, 2),
            "hi_52w": round(hi_52w, 2), "lo_52w": round(lo_52w, 2),
            "pct_from_52w_hi": round((latest / hi_52w - 1) * 100, 1) if hi_52w else 0,
            "ma20": round(ma20, 2) if ma20 is not None else "N/A",
            "ma50": round(ma50, 2) if ma50 is not None else "N/A",
            "ma200": round(ma200, 2) if ma200 is not None else "N/A",
            "above_ma20": (latest > ma20) if ma20 is not None else None,
            "above_ma50": (latest > ma50) if ma50 is not None else None,
            "above_ma200": (latest > ma200) if ma200 is not None else None,
            "rsi": round(rsi, 1) if rsi is not None else "N/A",
            "macd_hist": round(macd_hist, 3), "macd_bullish": macd_hist > 0,
            "vol_ratio": round(vol_ratio, 2), "bb_pct": round(bb_pct, 1),
            "pe_ratio": round(pe_ratio, 1) if pe_ratio else "N/A",
            "rev_growth": round(rev_growth, 1), "eps_growth": round(eps_growth, 1),
            "fcf_yield": round(fcf_yield, 1), "market_cap_b": round(market_cap_b, 1),
            "beta": round(beta, 2), "short_ratio": round(short_ratio, 1),
            "inst_pct": round(inst_pct, 1), "analyst_rec": round(analyst_rec, 2),
            "target_price": round(target_price, 2), "upside": round(upside, 1),
            "thin_history": thin_history, "trading_days": n_days,
        }
    except Exception as e:
        logger.error(f"{sym} 数据抓取失败 / data fetch failed: {e}")
        return _fallback(sym)


def _fallback(sym: str) -> dict:
    return {
        "sym": sym, "price": 0, "change_pct": 0, "hi_52w": 0, "lo_52w": 0, "pct_from_52w_hi": 0,
        "ma20": "N/A", "ma50": "N/A", "ma200": "N/A",
        "above_ma20": None, "above_ma50": None, "above_ma200": None,
        "rsi": "N/A", "macd_hist": 0, "macd_bullish": False, "vol_ratio": 1, "bb_pct": 50,
        "pe_ratio": "N/A", "rev_growth": 0, "eps_growth": 0, "fcf_yield": 0,
        "market_cap_b": 0, "beta": 1.0, "short_ratio": 0, "inst_pct": 0,
        "analyst_rec": 3, "target_price": 0, "upside": 0,
        "thin_history": True, "trading_days": 0, "error": True,
    }


# ── 宏观市场化指标：VIX / DXY / 美债 / 油价 / 黄金 / 大盘 ──────────────────────

def get_macro_market_snapshot() -> dict:
    """
    市场化宏观指标（无需额外API key，全部来自 yfinance）：
    VIX, DXY, US10Y, WTI原油, 黄金, 标普500, 纳指
    """
    tickers_map = {
        "vix":    "^VIX",
        "dxy":    "DX-Y.NYB",
        "us10y":  "^TNX",     # 10年期美债收益率
        "us2y":   "^IRX",     # 13周国库券，作短端利率代理
        "crude":  "CL=F",     # WTI原油期货
        "gold":   "GC=F",
        "sp500":  "^GSPC",
        "nasdaq": "^IXIC",
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

    if result.get("us10y") is not None:
        # 10Y-2Y 利差的简化代理 (此处用历史平均近似值作偏离参考，非精确期限利差)
        result["yield_spread"] = round(result["us10y"] - 4.5, 2)

    return result


# ── 宏观经济数据：CPI / NFP / PMI / Fed Funds Rate (通过 FRED API，可选) ──────

FRED_SERIES = {
    "cpi_yoy":       "CPIAUCSL",   # CPI 城镇消费者价格指数（计算环比需两期数据）
    "core_pce_yoy":  "PCEPILFE",   # 核心PCE（Fed更看重的通胀指标）
    "nfp":           "PAYEMS",     # 非农就业人数（千人，需计算环比变化）
    "unemployment":  "UNRATE",     # 失业率
    "fed_funds":     "FEDFUNDS",   # 联邦基金利率
}


def _fred_latest_two(series_id: str) -> tuple:
    """从 FRED 获取某个序列最近两期数据，返回 (最新值, 上一期值, 日期)。"""
    if not FRED_API_KEY:
        return None, None, None
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id, "api_key": FRED_API_KEY,
            "file_type": "json", "sort_order": "desc", "limit": 14,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        obs = [o for o in data.get("observations", []) if o["value"] not in (".", "")]
        if len(obs) < 2:
            return None, None, None
        latest, prior = obs[0], obs[1]
        return float(latest["value"]), float(prior["value"]), latest["date"]
    except Exception as e:
        logger.warning(f"FRED 数据获取失败 {series_id}: {e}")
        return None, None, None


def get_macro_economic_data() -> dict:
    """
    抓取 CPI / NFP / 失业率 / Fed Funds Rate。
    若未配置 FRED_API_KEY，返回空字典（AI解读时会基于市场化指标做定性判断）。
    """
    if not FRED_API_KEY:
        logger.info("未配置 FRED_API_KEY，跳过CPI/NFP宏观经济数据抓取（不影响其他维度）")
        return {}

    econ = {}
    try:
        cpi_now, cpi_prev, cpi_date = _fred_latest_two(FRED_SERIES["cpi_yoy"])
        if cpi_now and cpi_prev:
            econ["cpi_mom_pct"] = round((cpi_now - cpi_prev) / cpi_prev * 100, 2)
            econ["cpi_date"] = cpi_date

        pce_now, pce_prev, pce_date = _fred_latest_two(FRED_SERIES["core_pce_yoy"])
        if pce_now and pce_prev:
            econ["core_pce_mom_pct"] = round((pce_now - pce_prev) / pce_prev * 100, 2)

        nfp_now, nfp_prev, nfp_date = _fred_latest_two(FRED_SERIES["nfp"])
        if nfp_now and nfp_prev:
            econ["nfp_change_k"] = round(nfp_now - nfp_prev, 0)
            econ["nfp_date"] = nfp_date

        unemp_now, unemp_prev, _ = _fred_latest_two(FRED_SERIES["unemployment"])
        if unemp_now is not None:
            econ["unemployment_rate"] = unemp_now
            econ["unemployment_chg"] = round(unemp_now - (unemp_prev or unemp_now), 2)

        fed_now, fed_prev, fed_date = _fred_latest_two(FRED_SERIES["fed_funds"])
        if fed_now is not None:
            econ["fed_funds_rate"] = fed_now
            econ["fed_funds_chg"] = round(fed_now - (fed_prev or fed_now), 2)

    except Exception as e:
        logger.warning(f"宏观经济数据处理出错: {e}")

    return econ


def get_macro_snapshot() -> dict:
    """合并市场化指标 + 经济数据，构成完整宏观快照。"""
    snap = get_macro_market_snapshot()
    snap.update(get_macro_economic_data())
    return snap


def vix_regime(vix) -> tuple:
    """
    VIX 水平含义解读层 — 不只是数字，给出明确的市场情绪分类。
    返回 (中文描述, 英文描述, 风险调整分数)
    """
    if vix is None:
        return "数据不可用", "Data unavailable", 0
    if vix < 13:
        return ("极度平静 — 可能隐含市场自满情绪", "Extreme calm — may signal complacency", -3)
    if vix < 17:
        return ("低恐慌 — 风险偏好健康", "Low fear — healthy risk appetite", 8)
    if vix < 22:
        return ("正常波动区间", "Normal volatility range", 3)
    if vix < 28:
        return ("波动加剧 — 市场开始定价不确定性", "Elevated — market pricing in uncertainty", -8)
    if vix < 35:
        return ("高度紧张 — 避险情绪主导", "High stress — risk-off dominant", -15)
    return ("极端恐慌 — 类似危机模式", "Extreme panic — crisis-like conditions", -22)


# ════════════════════════════════════════════════════════════════════════════
#  2. MFTSR 五维规则评分 — Scoring Engine
# ════════════════════════════════════════════════════════════════════════════

def score_macro(macro: dict) -> tuple:
    """
    宏观维度评分，综合：
      - 经济数据 CPI/NFP/失业率方向
      - 美联储动态 Fed Funds Rate 变化
      - VIX 水平及含义（解读层，非仅数字）
      - 美元指数 DXY
      - 油价 WTI 走势
      - 美债收益率 US10Y
      - 大盘（纳指）动量
    （地缘政治/关税风险定性部分在 AI commentary 中处理，规则引擎无法量化突发新闻）
    """
    score, sig = 50, []

    # ── VIX 解读层 ──────────────────────────────────────
    vix = macro.get("vix")
    if vix is not None:
        desc_zh, desc_en, adj = vix_regime(vix)
        score += adj
        sig.append(f"VIX={vix:.1f} → {desc_zh} ({desc_en})")

    # ── 美联储动态 Fed Funds Rate ───────────────────────
    fed_chg = macro.get("fed_funds_chg")
    fed_rate = macro.get("fed_funds_rate")
    if fed_rate is not None:
        if fed_chg and fed_chg > 0.05:
            score -= 10; sig.append(f"美联储加息中 Fed={fed_rate:.2f}% (+{fed_chg:.2f}) 鹰派压制估值")
        elif fed_chg and fed_chg < -0.05:
            score += 10; sig.append(f"美联储降息中 Fed={fed_rate:.2f}% ({fed_chg:.2f}) 利好风险资产")
        else:
            sig.append(f"Fed Funds Rate维持 {fed_rate:.2f}%，政策按兵不动")

    # ── 通胀数据 CPI / 核心PCE ──────────────────────────
    cpi_mom = macro.get("cpi_mom_pct")
    if cpi_mom is not None:
        if cpi_mom > 0.4:
            score -= 8; sig.append(f"CPI环比+{cpi_mom:.2f}% 通胀升温，加息压力增加")
        elif cpi_mom < 0.1:
            score += 6; sig.append(f"CPI环比+{cpi_mom:.2f}% 通胀降温，利好降息预期")
        else:
            sig.append(f"CPI环比+{cpi_mom:.2f}% 温和")

    # ── 就业数据 NFP / 失业率 ───────────────────────────
    nfp_chg = macro.get("nfp_change_k")
    if nfp_chg is not None:
        if nfp_chg > 200:
            sig.append(f"非农新增{nfp_chg:.0f}千人，劳动力市场强劲（可能延后降息）")
        elif nfp_chg < 100:
            score -= 5; sig.append(f"非农新增仅{nfp_chg:.0f}千人，就业市场降温")

    unemp = macro.get("unemployment_rate")
    if unemp is not None:
        chg = macro.get("unemployment_chg", 0)
        if chg > 0.2:
            score -= 6; sig.append(f"失业率{unemp:.1f}% 上升中 ⚠️ 经济放缓信号")

    # ── 美元指数 DXY ────────────────────────────────────
    dxy_chg = macro.get("dxy_chg_pct")
    dxy = macro.get("dxy")
    if dxy_chg is not None:
        if dxy_chg > 0.5:   score -= 6; sig.append(f"美元走强 DXY={dxy} (+{dxy_chg:.2f}%) 压制风险资产")
        elif dxy_chg < -0.5: score += 6; sig.append(f"美元走弱 DXY={dxy} ({dxy_chg:.2f}%) 利好风险资产")

    # ── 油价 WTI ────────────────────────────────────────
    crude_chg = macro.get("crude_chg_pct")
    crude = macro.get("crude")
    if crude_chg is not None:
        if crude_chg > 3:
            score -= 5; sig.append(f"油价跳涨 WTI=${crude} (+{crude_chg:.2f}%) 地缘风险/通胀压力")
        elif crude_chg < -3:
            score += 3; sig.append(f"油价回落 WTI=${crude} ({crude_chg:.2f}%) 通胀压力缓解")
        else:
            sig.append(f"WTI原油 ${crude} ({crude_chg:+.2f}%)")

    # ── 美债收益率 US10Y ────────────────────────────────
    us10y = macro.get("us10y")
    if us10y is not None:
        sig.append(f"US10Y={us10y:.2f}%")
        spread = macro.get("yield_spread")
        if spread is not None and spread < -0.3:
            score -= 8; sig.append(f"收益率曲线倒挂{spread:.2f}% ⚠️ 衰退预警信号")

    # ── 大盘动量（纳指）──────────────────────────────────
    nq_chg = macro.get("nasdaq_chg_pct")
    if nq_chg is not None:
        if nq_chg > 0.5:    score += 5; sig.append(f"纳指+{nq_chg:.2f}% 风险偏好积极")
        elif nq_chg < -1.0: score -= 8; sig.append(f"纳指{nq_chg:.2f}% 避险情绪升温")

    return max(0, min(100, score)), sig[:7]


def score_fundamental(md: dict) -> tuple:
    score, sig = 50, []
    pe = md.get("pe_ratio")
    if pe and pe != "N/A":
        if pe < 15:   score += 12; sig.append(f"PE={pe}x 估值便宜")
        elif pe < 25: score += 6;  sig.append(f"PE={pe}x 估值合理")
        elif pe < 40: score -= 3;  sig.append(f"PE={pe}x 估值偏高")
        else:         score -= 10; sig.append(f"PE={pe}x 估值昂贵")

    rev = md.get("rev_growth", 0)
    if rev > 25:    score += 14; sig.append(f"营收增速+{rev:.1f}% 超高增长")
    elif rev > 10:  score += 7;  sig.append(f"营收增速+{rev:.1f}% 健康")
    elif rev > 0:   score += 2;  sig.append(f"营收增速+{rev:.1f}% 温和正增长")
    elif rev > -5:  score -= 6;  sig.append(f"营收增速{rev:.1f}% 轻微负增长")
    else:           score -= 14; sig.append(f"营收下滑{rev:.1f}% ⚠️")

    eps = md.get("eps_growth", 0)
    if eps > 20:    score += 8; sig.append(f"EPS增速+{eps:.1f}%")
    elif eps < -10: score -= 10; sig.append(f"EPS下滑{eps:.1f}% ⚠️")

    fcf = md.get("fcf_yield", 0)
    if fcf > 5: score += 8; sig.append(f"FCF收益率{fcf:.1f}% 优秀")
    elif fcf < 0: score -= 6; sig.append(f"FCF为负{fcf:.1f}%")

    upside = md.get("upside", 0)
    if upside > 20: score += 6; sig.append(f"目标价上行空间+{upside:.1f}%")
    elif upside < -10: score -= 6; sig.append(f"目标价低于现价{upside:.1f}%")

    return max(0, min(100, score)), sig[:4]


def score_technical(md: dict) -> tuple:
    # 新上市股票（如SPCX）数据不足时，技术面无法可靠评分，直接返回中性分并标注
    if md.get("thin_history"):
        days = md.get("trading_days", 0)
        return 50, [f"⚠️ 上市仅{days}个交易日，技术面指标(MA/RSI)数据不足，暂不参与评分（中性处理）"]

    score, sig = 50, []
    above_ma20  = md.get("above_ma20")
    above_ma50  = md.get("above_ma50")
    above_ma200 = md.get("above_ma200")
    ma_flags = [f for f in [above_ma20, above_ma50, above_ma200] if f is not None]
    ma_count = sum(1 for f in ma_flags if f)

    if len(ma_flags) == 3:
        if ma_count == 3:   score += 15; sig.append("多头排列：站上MA20/50/200 ✅")
        elif ma_count == 2: score += 7;  sig.append("中性偏多")
        elif ma_count == 1: score -= 7;  sig.append("中性偏空")
        else:                score -= 15; sig.append("空头排列 ⚠️")
    elif ma_flags:
        sig.append(f"均线数据部分可用（{ma_count}/{len(ma_flags)}条上方）")

    rsi = md.get("rsi", "N/A")
    if isinstance(rsi, (int, float)):
        if rsi < 30:   score += 10; sig.append(f"RSI={rsi:.0f} 超卖，反弹概率高")
        elif rsi > 75: score -= 12; sig.append(f"RSI={rsi:.0f} 严重超买")
        elif rsi > 65: score -= 5;  sig.append(f"RSI={rsi:.0f} 偏热")

    macd_hist = md.get("macd_hist", 0)
    if macd_hist > 0.1:  score += 10; sig.append(f"MACD柱+{macd_hist:.3f} 动量向上")
    elif macd_hist < -0.1: score -= 10; sig.append(f"MACD柱{macd_hist:.3f} 动量向下")

    vol_ratio, chg = md.get("vol_ratio", 1), md.get("change_pct", 0)
    if vol_ratio > 1.5 and chg > 1:   score += 8; sig.append(f"放量上涨{vol_ratio:.1f}x")
    elif vol_ratio > 1.5 and chg < -1: score -= 8; sig.append(f"放量下跌{vol_ratio:.1f}x")

    bb_pct = md.get("bb_pct", 50)
    if bb_pct > 90:  score -= 8; sig.append(f"布林上轨附近，短期过热")
    elif bb_pct < 10: score += 8; sig.append(f"布林下轨附近，超卖反弹机会")

    return max(0, min(100, score)), sig[:4]


def score_sentiment(md: dict) -> tuple:
    score, sig = 50, []
    inst = md.get("inst_pct", 0)
    if inst > 75: score += 8; sig.append(f"机构持仓{inst:.1f}% 高")
    elif inst < 20: score -= 5; sig.append(f"机构持仓{inst:.1f}% 偏低")

    short = md.get("short_ratio", 0)
    if short < 2: score += 5; sig.append(f"空头比率{short:.1f} 压力小")
    elif short > 5: score -= 8; sig.append(f"空头比率{short:.1f} ⚠️ 空头聚集")

    rec = md.get("analyst_rec", 3)
    if rec <= 1.8: score += 12; sig.append(f"分析师评级{rec:.1f} 强烈买入")
    elif rec >= 3.5: score -= 8; sig.append(f"分析师评级{rec:.1f} 偏向卖出")

    upside = md.get("upside", 0)
    if upside > 25: score += 8; sig.append(f"目标价空间+{upside:.1f}% 分析师看多")
    elif upside < -5: score -= 8; sig.append(f"目标价已低于现价")

    return max(0, min(100, score)), sig[:4]


def score_risk(md: dict, macro: dict) -> tuple:
    score, sig = 60, []
    beta = md.get("beta", 1.0)
    if beta < 0.7: score += 10; sig.append(f"Beta={beta:.2f} 低波动")
    elif beta < 1.2: score += 3; sig.append(f"Beta={beta:.2f} 市场中性")
    elif beta < 1.8: score -= 8; sig.append(f"Beta={beta:.2f} 高波动")
    else: score -= 18; sig.append(f"Beta={beta:.2f} ⚠️ 极高波动")

    pct_from_hi = md.get("pct_from_52w_hi", 0)
    if pct_from_hi > -5: score -= 8; sig.append(f"距52周高点{pct_from_hi:.1f}% 高位风险")
    elif pct_from_hi < -30: score += 8; sig.append(f"距52周高点{pct_from_hi:.1f}% 回撤充分")

    vix = macro.get("vix")
    if vix is not None:
        if vix > 30: score -= 15; sig.append(f"VIX={vix:.1f} ⚠️ 系统性风险高")
        elif vix > 22: score -= 6; sig.append(f"VIX={vix:.1f} 适度谨慎")

    cap = md.get("market_cap_b", 0)
    if cap > 200: score += 5; sig.append(f"市值${cap:.0f}B 流动性好")
    elif 0 < cap < 10: score -= 8; sig.append(f"市值${cap:.1f}B 流动性风险较高")

    # 油价/地缘政治敞口的简化风险代理（高Beta + 高油价波动 = 额外风险）
    crude_chg = macro.get("crude_chg_pct")
    if crude_chg is not None and abs(crude_chg) > 4:
        score -= 4; sig.append(f"油价剧烈波动({crude_chg:+.1f}%)，地缘政治风险溢价上升")

    return max(0, min(100, score)), sig[:5]


def compute_mftsr(md: dict, macro: dict) -> dict:
    m, m_sig = score_macro(macro)
    f, f_sig = score_fundamental(md)
    t, t_sig = score_technical(md)
    s, s_sig = score_sentiment(md)
    r, r_sig = score_risk(md, macro)

    composite = round(m * WEIGHTS["macro"] + f * WEIGHTS["fundamental"] +
                       t * WEIGHTS["technical"] + s * WEIGHTS["sentiment"] +
                       r * WEIGHTS["risk"])

    if composite >= SCORE_BUY:   signal, action = "买入信号", "BUY"
    elif composite >= SCORE_HOLD: signal, action = "持有观望", "HOLD"
    else:                          signal, action = "减仓警示", "SELL"

    dims = {"macro": m, "fundamental": f, "technical": t, "sentiment": s, "risk": r}
    alerts = [k for k, v in dims.items() if v < ALERT_DIM]

    price = md.get("price", 0)
    stop_pct = 0.06 if md.get("beta", 1) > 1.5 else 0.04
    stop_loss = round(price * (1 - stop_pct), 2)
    edge = (composite - 50) / 50
    position_pct = max(0, min(20, round(edge * 15)))

    return {
        "composite": composite, "signal": signal, "action": action,
        "macro": m, "fundamental": f, "technical": t, "sentiment": s, "risk": r,
        "macro_sig": m_sig, "fundamental_sig": f_sig, "technical_sig": t_sig,
        "sentiment_sig": s_sig, "risk_sig": r_sig,
        "alerts": alerts, "stop_loss": stop_loss, "position_pct": position_pct,
    }


# ════════════════════════════════════════════════════════════════════════════
#  3. Claude AI 文字解读 — Bilingual Commentary
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_ZH = """你是MFTSR Alpha系统的首席分析师，融合了：
- Ray Dalio的宏观债务周期思维
- Peter Lynch基本面精选股票哲学
- Stanley Druckenmiller趋势跟随与风险管理
- Jim Simons量化信号体系

你必须在宏观分析中明确覆盖：经济数据（CPI/非农/PMI走向）、美联储政策动态、
地缘政治风险（如伊朗/中东局势、关税政策对相关行业供应链和成本的影响）、
VIX水平的市场情绪含义、油价与美债收益率走势。

输出将直接发送到Telegram。风格要求：直接、精准、专业，禁止废话；
数字和事实优先；不确定时明确说"信号混合，等待确认"；篇幅紧凑；用简体中文输出。"""

SYSTEM_PROMPT_EN = """You are the Chief Analyst of the MFTSR Alpha system, combining:
- Ray Dalio's macro debt-cycle framework
- Peter Lynch's fundamentals-driven stock-picking philosophy
- Stanley Druckenmiller's trend-following and risk management
- Jim Simons' quantitative signal discipline

Your macro analysis MUST explicitly cover: economic data (CPI/NFP/PMI trends),
Federal Reserve policy stance, geopolitical risk (e.g. Iran/Middle East tensions,
tariff policy impact on relevant supply chains and input costs), what the current
VIX level means for market sentiment, and oil price / Treasury yield trends.

Output goes directly to Telegram. Style: direct, precise, professional, no filler;
numbers and facts first; say "mixed signals, awaiting confirmation" when genuinely
uncertain rather than forcing a call; keep it tight; output in English."""


def _macro_context_str(macro: dict, lang: str) -> str:
    vix = macro.get('vix', 'N/A')
    vix_desc_zh, vix_desc_en, _ = vix_regime(vix if isinstance(vix, (int, float)) else None)
    vix_desc = vix_desc_zh if lang == "zh" else vix_desc_en

    if lang == "zh":
        parts = [
            f"VIX={vix} ({vix_desc})",
            f"DXY={macro.get('dxy','N/A')}",
            f"US10Y={macro.get('us10y','N/A')}%",
            f"WTI原油=${macro.get('crude','N/A')} ({macro.get('crude_chg_pct',0):+.2f}%)",
            f"纳指={macro.get('nasdaq_chg_pct',0):+.2f}%",
        ]
        if macro.get("fed_funds_rate") is not None:
            parts.append(f"Fed Funds={macro['fed_funds_rate']:.2f}%")
        if macro.get("cpi_mom_pct") is not None:
            parts.append(f"CPI环比={macro['cpi_mom_pct']:+.2f}%")
        if macro.get("nfp_change_k") is not None:
            parts.append(f"非农变化={macro['nfp_change_k']:.0f}千人")
        if macro.get("unemployment_rate") is not None:
            parts.append(f"失业率={macro['unemployment_rate']:.1f}%")
    else:
        parts = [
            f"VIX={vix} ({vix_desc})",
            f"DXY={macro.get('dxy','N/A')}",
            f"US10Y={macro.get('us10y','N/A')}%",
            f"WTI Crude=${macro.get('crude','N/A')} ({macro.get('crude_chg_pct',0):+.2f}%)",
            f"Nasdaq={macro.get('nasdaq_chg_pct',0):+.2f}%",
        ]
        if macro.get("fed_funds_rate") is not None:
            parts.append(f"Fed Funds={macro['fed_funds_rate']:.2f}%")
        if macro.get("cpi_mom_pct") is not None:
            parts.append(f"CPI MoM={macro['cpi_mom_pct']:+.2f}%")
        if macro.get("nfp_change_k") is not None:
            parts.append(f"NFP Δ={macro['nfp_change_k']:.0f}k")
        if macro.get("unemployment_rate") is not None:
            parts.append(f"Unemployment={macro['unemployment_rate']:.1f}%")

    return "  ".join(parts)


def _fmt(val, suffix=""):
    """格式化可能是数字或'N/A'字符串的字段，避免格式化崩溃。"""
    if isinstance(val, (int, float)):
        return f"{val:.0f}{suffix}" if suffix == "" else f"{val:.1f}{suffix}"
    return str(val)


def build_prompt(ticker: dict, md: dict, scores: dict, macro: dict, report_type: str, lang: str) -> str:
    sym, name = ticker["sym"], ticker["name"]
    macro_str = _macro_context_str(macro, lang)
    rsi_str   = _fmt(md.get('rsi', 'N/A'))
    ma20_str  = md.get('ma20', 'N/A')
    ma50_str  = md.get('ma50', 'N/A')
    ma200_str = md.get('ma200', 'N/A')
    thin_note_zh = ""
    thin_note_en = ""
    if md.get("thin_history"):
        days = md.get("trading_days", 0)
        thin_note_zh = f"\n⚠️ 注意：{sym}上市仅{days}个交易日，长周期技术指标(均线/RSI)数据不足，技术面评分为中性占位值，请在分析中明确说明这一点，避免给出虚假的趋势判断"
        thin_note_en = f"\n⚠️ Note: {sym} has only {days} trading days of history post-IPO. Long-window technical indicators (MAs/RSI) are insufficient and the technical score is a neutral placeholder — explicitly flag this in your analysis rather than asserting a trend"

    if lang == "zh":
        label = REPORT_LABELS_ZH.get(report_type, "分析报告")
        alerts_str = f"\n⚠️ 风险警报维度（评分<30）: {', '.join(scores['alerts'])}" if scores.get("alerts") else ""
        fred_note = "" if FRED_API_KEY else "\n（注：CPI/NFP详细数据未配置FRED API，请基于市场指标定性判断宏观环境，并提示用户当前缺少官方经济数据源）"
        return f"""【{label}】{sym} — {name}

═══ 市场数据 ═══
价格: ${md['price']} ({md['change_pct']:+.2f}%)
52周区间: ${md['lo_52w']} — ${md['hi_52w']} (距高点{md['pct_from_52w_hi']:.1f}%)
MA20/50/200: ${ma20_str} / ${ma50_str} / ${ma200_str}
RSI(14)={rsi_str}  MACD柱={md['macd_hist']:.3f}  成交量比={md['vol_ratio']:.1f}x

═══ 基本面 ═══
PE={md['pe_ratio']}x  FCF收益率={md['fcf_yield']:.1f}%
营收增速={md['rev_growth']:+.1f}%  EPS增速={md['eps_growth']:+.1f}%
目标价=${md['target_price']} (上行空间{md['upside']:+.1f}%)

═══ MFTSR评分 ═══
综合: {scores['composite']}/100 → {scores['signal']}
宏观{scores['macro']} 基本面{scores['fundamental']} 技术面{scores['technical']} 情绪{scores['sentiment']} 风险{scores['risk']}{alerts_str}

═══ 宏观环境（含经济数据/Fed/地缘政治背景需你补充解读）═══
{macro_str}{fred_note}

请严格按以下结构输出（不加多余章节），全部用简体中文：

【核心判断】一句话总结
【宏观面】（3点：①经济数据CPI/NFP/PMI方向 ②美联储政策动态 ③地缘政治/关税风险，如伊朗中东局势对油价和相关板块的潜在影响——基于你的知识给出合理评估，并说明这是定性判断而非实时新闻）
【基本面】（2点，估值与成长性评估）
【技术面】（2点，趋势与关键价位）
【情绪与资金】（1-2点）
【风险提示】（1-2个具体风险点）
【操作建议】
• 信号: {scores['action']}
• 止损参考: ${scores['stop_loss']}
• 仓位建议: 不超过{scores['position_pct']}%
• 近期关注: [本周关键催化剂，包括可能的经济数据公布或地缘政治事件]"""

    else:  # English
        label = REPORT_LABELS_EN.get(report_type, "Analysis Report")
        alerts_str = f"\n⚠️ Risk Alert (score<30): {', '.join(scores['alerts'])}" if scores.get("alerts") else ""
        fred_note = "" if FRED_API_KEY else "\n(Note: detailed CPI/NFP data not configured via FRED API — provide qualitative macro assessment based on market-based indicators, and flag that the official economic data feed is not connected)"
        return f"""[{label}] {sym} — {name}

=== Market Data ===
Price: ${md['price']} ({md['change_pct']:+.2f}%)
52W Range: ${md['lo_52w']} — ${md['hi_52w']} ({md['pct_from_52w_hi']:.1f}% from high)
MA20/50/200: ${ma20_str} / ${ma50_str} / ${ma200_str}
RSI(14)={rsi_str}  MACD hist={md['macd_hist']:.3f}  Volume ratio={md['vol_ratio']:.1f}x

=== Fundamentals ===
PE={md['pe_ratio']}x  FCF yield={md['fcf_yield']:.1f}%
Revenue growth={md['rev_growth']:+.1f}%  EPS growth={md['eps_growth']:+.1f}%
Target price=${md['target_price']} (upside {md['upside']:+.1f}%)

=== MFTSR Scores ===
Composite: {scores['composite']}/100 → {scores['signal']}
Macro {scores['macro']}  Fundamental {scores['fundamental']}  Technical {scores['technical']}  Sentiment {scores['sentiment']}  Risk {scores['risk']}{alerts_str}

=== Macro Context (economic data / Fed / geopolitical backdrop — add your interpretation) ===
{macro_str}{fred_note}

Output strictly in this structure (no extra sections), entirely in English:

[Core Call] One-sentence summary
[Macro] (3 points: (1) CPI/NFP/PMI direction (2) Fed policy stance (3) Geopolitical/tariff risk — e.g. Iran/Middle East tensions and their potential impact on oil prices and relevant sectors, based on your knowledge; clearly flag this as qualitative judgment, not real-time news)
[Fundamentals] (2 points on valuation and growth quality)
[Technical] (2 points on trend and key levels)
[Sentiment & Flows] (1-2 points)
[Risk Flags] (1-2 specific risks)
[Action]
- Signal: {scores['action']}
- Stop-loss reference: ${scores['stop_loss']}
- Position sizing: max {scores['position_pct']}% of portfolio
- Watch this week: [key catalysts, including possible economic data releases or geopolitical events]"""


def get_ai_commentary(ticker: dict, md: dict, scores: dict, macro: dict, report_type: str, lang: str) -> str:
    if not ANTHROPIC_KEY:
        return _fallback_commentary(ticker, md, scores, lang)
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        prompt = build_prompt(ticker, md, scores, macro, report_type, lang)
        system = SYSTEM_PROMPT_ZH if lang == "zh" else SYSTEM_PROMPT_EN
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1000,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.error(f"Claude API 调用失败 {ticker['sym']} [{lang}]: {e}")
        return _fallback_commentary(ticker, md, scores, lang)


def _fallback_commentary(ticker: dict, md: dict, scores: dict, lang: str) -> str:
    sym = ticker["sym"]
    rsi_disp = _fmt(md.get('rsi', 'N/A'))
    ma50_status_zh = '多头排列' if md.get('above_ma50') else ('跌破MA50' if md.get('above_ma50') is False else '数据不足')
    ma50_status_en = 'above MA50' if md.get('above_ma50') else ('below MA50' if md.get('above_ma50') is False else 'insufficient data')
    if lang == "zh":
        return (f"[AI离线] {sym} 规则评分摘要\n综合评分: {scores['composite']}/100 → {scores['signal']}\n"
                f"RSI={rsi_disp}, {ma50_status_zh}\n"
                f"止损${scores['stop_loss']} | 仓位≤{scores['position_pct']}%")
    return (f"[AI offline] {sym} rule-based summary\nComposite: {scores['composite']}/100 → {scores['signal']}\n"
            f"RSI={rsi_disp}, {ma50_status_en}\n"
            f"Stop-loss ${scores['stop_loss']} | Position ≤{scores['position_pct']}%")


# ════════════════════════════════════════════════════════════════════════════
#  4. Telegram 发送 — Bilingual Dual Messages
# ════════════════════════════════════════════════════════════════════════════

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def _send(text: str) -> bool:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.error("Telegram 未配置，跳过发送 / not configured, skipping send")
        return False
    try:
        resp = requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=15)
        data = resp.json()
        if not data.get("ok"):
            logger.error(f"Telegram 错误 / error: {data}")
            return False
        return True
    except requests.RequestException as e:
        logger.error(f"Telegram 网络错误 / network error: {e}")
        return False


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _score_bar(score: int, width: int = 8) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _signal_emoji(action: str) -> str:
    return {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(action, "⚪")


def _dim_label(score: int) -> str:
    if score >= 70: return "🔋🔋🔋"
    if score >= 50: return "🔋🔋░"
    if score >= 30: return "🔋░░"
    return "⚠️░░"


def build_header(report_type: str, macro: dict, lang: str) -> str:
    labels = REPORT_LABELS_ZH if lang == "zh" else REPORT_LABELS_EN
    label = labels.get(report_type, "MFTSR Report")
    uk_now = datetime.now().strftime("%Y-%m-%d %H:%M")

    vix = macro.get("vix", "N/A")
    vix_chg = macro.get("vix_chg_pct", 0)
    vix_icon = "😱" if isinstance(vix, (int, float)) and vix > 25 else ("😰" if isinstance(vix, (int, float)) and vix > 18 else "😌")
    vix_desc_zh, vix_desc_en, _ = vix_regime(vix if isinstance(vix, (int, float)) else None)
    vix_desc = vix_desc_zh if lang == "zh" else vix_desc_en

    if lang == "zh":
        macro_lines = [
            f"  {vix_icon} VIX: <b>{vix}</b> ({vix_chg:+.2f}%) — {vix_desc}",
            f"  💵 DXY: <b>{macro.get('dxy','N/A')}</b>",
            f"  📈 S&amp;P500: <b>{macro.get('sp500_chg_pct',0):+.2f}%</b>  纳指: <b>{macro.get('nasdaq_chg_pct',0):+.2f}%</b>",
            f"  🏦 US10Y: <b>{macro.get('us10y','N/A')}%</b>",
            f"  🛢 WTI原油: <b>${macro.get('crude','N/A')}</b> ({macro.get('crude_chg_pct',0):+.2f}%)",
            f"  🥇 黄金: <b>{macro.get('gold','N/A')}</b>",
        ]
        if macro.get("fed_funds_rate") is not None:
            macro_lines.append(f"  🏛 Fed Funds Rate: <b>{macro['fed_funds_rate']:.2f}%</b>")
        if macro.get("cpi_mom_pct") is not None:
            macro_lines.append(f"  📊 CPI环比: <b>{macro['cpi_mom_pct']:+.2f}%</b>")
        if macro.get("nfp_change_k") is not None:
            macro_lines.append(f"  👷 非农变化: <b>{macro['nfp_change_k']:+.0f}k</b>")
        title = "🌐 宏观环境快照"
    else:
        macro_lines = [
            f"  {vix_icon} VIX: <b>{vix}</b> ({vix_chg:+.2f}%) — {vix_desc}",
            f"  💵 DXY: <b>{macro.get('dxy','N/A')}</b>",
            f"  📈 S&amp;P500: <b>{macro.get('sp500_chg_pct',0):+.2f}%</b>  Nasdaq: <b>{macro.get('nasdaq_chg_pct',0):+.2f}%</b>",
            f"  🏦 US10Y: <b>{macro.get('us10y','N/A')}%</b>",
            f"  🛢 WTI Crude: <b>${macro.get('crude','N/A')}</b> ({macro.get('crude_chg_pct',0):+.2f}%)",
            f"  🥇 Gold: <b>{macro.get('gold','N/A')}</b>",
        ]
        if macro.get("fed_funds_rate") is not None:
            macro_lines.append(f"  🏛 Fed Funds Rate: <b>{macro['fed_funds_rate']:.2f}%</b>")
        if macro.get("cpi_mom_pct") is not None:
            macro_lines.append(f"  📊 CPI MoM: <b>{macro['cpi_mom_pct']:+.2f}%</b>")
        if macro.get("nfp_change_k") is not None:
            macro_lines.append(f"  👷 NFP Change: <b>{macro['nfp_change_k']:+.0f}k</b>")
        title = "🌐 Macro Snapshot"

    time_label = "(UK时间)" if lang == "zh" else "(UK time)"
    return (
        f"<b>◈ MFTSR ALPHA — {label}</b>\n"
        f"🕐 {uk_now} {time_label}\n"
        f"{'─'*32}\n"
        f"<b>{title}</b>\n"
        + "\n".join(macro_lines) + "\n"
        f"{'─'*32}"
    )


def build_ticker_message(ticker: dict, md: dict, scores: dict, commentary: str, lang: str) -> str:
    sym, name = ticker["sym"], ticker["name"]
    chg_arrow = "▲" if md["change_pct"] >= 0 else "▼"
    sig = _signal_emoji(scores["action"])

    if lang == "zh":
        alert_label = "风险警报"
        dim_labels = ["宏观", "基本面", "技术面", "情绪", "风险"]
        composite_label = "MFTSR综合"
        stop_label, pos_label = "止损", "仓位"
    else:
        alert_label = "Risk Alert"
        dim_labels = ["Macro", "Fundamental", "Technical", "Sentiment", "Risk"]
        composite_label = "MFTSR Composite"
        stop_label, pos_label = "Stop-loss", "Position"

    alert_line = f"\n⚠️ <b>{alert_label}</b>: {', '.join(scores['alerts'])}" if scores.get("alerts") else ""
    rsi_disp = _fmt(md.get('rsi', 'N/A'))

    return (
        f"\n{sig} <b>{sym}</b>  {_escape(name)}\n"
        f"  💰 <b>${md['price']}</b>  {chg_arrow}{abs(md['change_pct']):.2f}%   "
        f"RSI={rsi_disp}  Vol={md['vol_ratio']:.1f}x\n"
        f"\n<b>{composite_label}: {scores['composite']}/100</b>  [{_score_bar(scores['composite'])}]\n"
        f"  {dim_labels[0]}{scores['macro']:>3} {_dim_label(scores['macro'])}  "
        f"{dim_labels[1]}{scores['fundamental']:>3} {_dim_label(scores['fundamental'])}\n"
        f"  {dim_labels[2]}{scores['technical']:>3} {_dim_label(scores['technical'])}  "
        f"{dim_labels[3]}{scores['sentiment']:>3} {_dim_label(scores['sentiment'])}\n"
        f"  {dim_labels[4]}{scores['risk']:>3} {_dim_label(scores['risk'])}"
        f"{alert_line}\n"
        f"\n{'─'*28}\n{_escape(commentary)}\n{'─'*28}\n"
        f"🎯 <b>{scores['signal']}</b>  |  {stop_label} ${scores['stop_loss']}  |  {pos_label} ≤{scores['position_pct']}%\n"
    )


def build_footer(results: list, lang: str) -> str:
    if lang == "zh":
        header = "\n📋 <b>本次报告汇总</b>\n" + "─"*28 + "\n"
        disclaimer = "\n❓ 以上内容仅供参考，不构成投资建议"
    else:
        header = "\n📋 <b>Session Summary</b>\n" + "─"*28 + "\n"
        disclaimer = "\n❓ For reference only — not investment advice"

    rows = []
    for r in sorted(results, key=lambda x: x["scores"]["composite"], reverse=True):
        sig = _signal_emoji(r["scores"]["action"])
        rows.append(f"  {sig} <b>{r['ticker']['sym']:<8}</b>  {r['scores']['composite']:>3}/100  {r['md']['change_pct']:+.2f}%")
    return header + "\n".join(rows) + "\n" + "─"*28 + disclaimer


def send_report_lang(report_type: str, macro: dict, results: list, lang: str):
    """发送单一语言的完整报告（header + 每只股票 + footer）。"""
    _send(build_header(report_type, macro, lang))
    time.sleep(1)
    for r in results:
        msg = build_ticker_message(r["ticker"], r["md"], r["scores"], r["commentary"][lang], lang)
        if len(msg) > 4000:
            for i in range(0, len(msg), 3900):
                _send(msg[i:i+3900]); time.sleep(0.5)
        else:
            _send(msg)
        time.sleep(1.5)
    _send(build_footer(results, lang))
    logger.info(f"报告发送完成 [{lang}]: {report_type} — {len(results)}只股票")


# ════════════════════════════════════════════════════════════════════════════
#  5. 主流程 — Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info(f"开始生成报告 / Generating report: {REPORT_TYPE}")
    logger.info(f"股票池 / Watchlist: {[t['sym'] for t in WATCHLIST]}")

    macro = get_macro_snapshot()
    logger.info(
        f"宏观 / Macro: VIX={macro.get('vix')} DXY={macro.get('dxy')} "
        f"US10Y={macro.get('us10y')} Crude={macro.get('crude')} "
        f"Fed={macro.get('fed_funds_rate')}"
    )

    results = []
    for ticker in WATCHLIST:
        sym = ticker["sym"]
        logger.info(f"  分析 / Analysing {sym}...")
        try:
            md = get_market_data(sym)
            scores = compute_mftsr(md, macro)
            commentary_zh = get_ai_commentary(ticker, md, scores, macro, REPORT_TYPE, "zh")
            commentary_en = get_ai_commentary(ticker, md, scores, macro, REPORT_TYPE, "en")
            results.append({
                "ticker": ticker, "md": md, "scores": scores,
                "commentary": {"zh": commentary_zh, "en": commentary_en},
            })
            logger.info(f"  {sym}: composite={scores['composite']} ({scores['action']})")
        except Exception as e:
            logger.error(f"  {sym} 失败 / failed: {e}")
        time.sleep(2)

    if not results:
        logger.error("没有生成任何结果，终止发送 / No results generated, aborting")
        sys.exit(1)

    results.sort(key=lambda r: r["scores"]["composite"], reverse=True)
    top_results = results[:MAX_TICKERS]

    # 发送两条独立消息序列：先中文，后英文
    send_report_lang(REPORT_TYPE, macro, top_results, "zh")
    time.sleep(2)
    send_report_lang(REPORT_TYPE, macro, top_results, "en")

    logger.info("全部完成 / All done ✅")


if __name__ == "__main__":
    main()
