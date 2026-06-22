"""
a_share_report.py — A股早报 · UK时间 07:00 推送

报告结构：
  1. 隔夜美股动态（标普/纳指收盘表现）
  2. 宏观经济数据（CPI/非农/Fed动态，若配置FRED_API_KEY）
  3. VIX、美债收益率
  4. 地缘政治/关税风险定性（AI基于知识库判断）
  5. 中国国内政策面、消息面（AI基于知识库判断，明确标注非实时新闻）
  6. 持仓股+ETF+关注清单：实时价格、涨跌幅、技术面、操作建议

全文中文输出。

环境变量：
    ANTHROPIC_KEY, TELEGRAM_TOKEN, CHAT_ID — 必需
    FRED_API_KEY — 可选，用于美国宏观经济数据
"""

import os
import sys
import time
import logging
from datetime import datetime

import requests
import numpy as np
import yfinance as yf
import anthropic

from shared_config import A_SHARE_HOLDINGS, A_SHARE_WATCHLIST, A_SHARE_ALL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("a_share")

ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
FRED_API_KEY   = os.environ.get("FRED_API_KEY", "")

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ════════════════════════════════════════════════════════════════════════════
#  1. 隔夜美股 + 宏观数据
# ════════════════════════════════════════════════════════════════════════════

def get_us_overnight_snapshot() -> dict:
    """隔夜美股收盘表现 + VIX + 美债 + 油价 + 美元指数。"""
    tickers_map = {
        "sp500": "^GSPC", "nasdaq": "^IXIC", "dow": "^DJI",
        "vix": "^VIX", "us10y": "^TNX", "dxy": "DX-Y.NYB",
        "crude": "CL=F", "gold": "GC=F",
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
        except Exception as e:
            logger.warning(f"获取{sym}失败: {e}")
            result[key] = None
    return result


FRED_SERIES = {
    "cpi_yoy": "CPIAUCSL", "nfp": "PAYEMS",
    "unemployment": "UNRATE", "fed_funds": "FEDFUNDS",
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


def get_us_macro_economic() -> dict:
    if not FRED_API_KEY:
        return {}
    econ = {}
    try:
        cpi_now, cpi_prev, _ = _fred_latest_two(FRED_SERIES["cpi_yoy"])
        if cpi_now and cpi_prev:
            econ["cpi_mom_pct"] = round((cpi_now - cpi_prev) / cpi_prev * 100, 2)
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


def vix_regime_zh(vix) -> str:
    if vix is None: return "数据不可用"
    if vix < 13:  return "极度平静，警惕自满情绪"
    if vix < 17:  return "低恐慌，风险偏好健康"
    if vix < 22:  return "正常波动区间"
    if vix < 28:  return "波动加剧，市场定价不确定性"
    if vix < 35:  return "高度紧张，避险主导"
    return "极端恐慌，类似危机模式"


# ════════════════════════════════════════════════════════════════════════════
#  2. A股/ETF 实时数据抓取
# ════════════════════════════════════════════════════════════════════════════

def get_a_share_data(sym: str) -> dict:
    """
    抓取A股个股/ETF价格与技术指标。
    yfinance对A股支持有限，常有延迟15-20分钟或数据缺失 — 这是已知限制，
    缺失时标注"数据不可用"，不引入付费数据源。
    """
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="60d", interval="1d")
        if hist.empty:
            return {"sym": sym, "available": False}

        close = hist["Close"]
        volume = hist["Volume"]
        n_days = len(close)
        latest = float(close.iloc[-1])
        prev   = float(close.iloc[-2]) if n_days > 1 else latest
        change_pct = (latest - prev) / prev * 100 if prev else 0

        ma5  = float(close.rolling(5).mean().iloc[-1])  if n_days >= 5  else None
        ma20 = float(close.rolling(20).mean().iloc[-1]) if n_days >= 20 else None

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs   = gain / loss.replace(0, np.nan)
        rsi  = float(100 - (100 / (1 + rs)).iloc[-1]) if n_days >= 14 and not rs.isna().iloc[-1] else None

        avg_vol = float(volume.rolling(20).mean().iloc[-1]) if n_days >= 20 else float(volume.mean())
        vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

        return {
            "sym": sym, "available": True,
            "price": round(latest, 2), "change_pct": round(change_pct, 2),
            "ma5": round(ma5, 2) if ma5 is not None else "N/A",
            "ma20": round(ma20, 2) if ma20 is not None else "N/A",
            "above_ma5": (latest > ma5) if ma5 is not None else None,
            "above_ma20": (latest > ma20) if ma20 is not None else None,
            "rsi": round(rsi, 1) if rsi is not None else "N/A",
            "vol_ratio": round(vol_ratio, 2),
        }
    except Exception as e:
        logger.warning(f"{sym} A股数据获取失败: {e}")
        return {"sym": sym, "available": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════════════════
#  3. Claude AI 解读 — 宏观背景 + 中国政策面 + 持仓建议
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一位资深A股策略分析师，同时精通美股宏观对A股/港股的传导逻辑。
你的分析需要清晰区分：
1. 客观数据（隔夜美股收盘、VIX、美债收益率等）— 直接引用
2. 定性判断（地缘政治、中国政策动向、行业消息）— 基于你的知识给出合理评估，
   并必须明确说明这是定性判断、非实时新闻抓取，提醒用户结合最新公告核实

风格：直接、精准、专业；数字优先；不确定时说"信号混合"而非强行下结论；
篇幅紧凑；全部用简体中文。输出将直接发送到Telegram。"""


def build_macro_section_prompt(us_snap: dict, us_econ: dict) -> str:
    vix = us_snap.get("vix")
    vix_desc = vix_regime_zh(vix)

    econ_lines = []
    if us_econ.get("cpi_mom_pct") is not None:
        econ_lines.append(f"CPI环比: {us_econ['cpi_mom_pct']:+.2f}%")
    if us_econ.get("nfp_change_k") is not None:
        econ_lines.append(f"非农变化: {us_econ['nfp_change_k']:+.0f}千人")
    if us_econ.get("unemployment_rate") is not None:
        econ_lines.append(f"失业率: {us_econ['unemployment_rate']:.1f}%")
    if us_econ.get("fed_funds_rate") is not None:
        econ_lines.append(f"Fed Funds Rate: {us_econ['fed_funds_rate']:.2f}%")
    econ_str = "  ".join(econ_lines) if econ_lines else "（未配置FRED API，无官方经济数据，请基于市场指标定性判断）"

    return f"""请生成今日A股早报的【海外宏观背景】部分。

═══ 隔夜美股收盘 ═══
标普500: {us_snap.get('sp500','N/A')} ({us_snap.get('sp500_chg_pct',0):+.2f}%)
纳斯达克: {us_snap.get('nasdaq','N/A')} ({us_snap.get('nasdaq_chg_pct',0):+.2f}%)
道指: {us_snap.get('dow','N/A')} ({us_snap.get('dow_chg_pct',0):+.2f}%)

═══ 风险指标 ═══
VIX: {vix} — {vix_desc}
美债10年期收益率: {us_snap.get('us10y','N/A')}%
美元指数DXY: {us_snap.get('dxy','N/A')} ({us_snap.get('dxy_chg_pct',0):+.2f}%)
WTI原油: ${us_snap.get('crude','N/A')} ({us_snap.get('crude_chg_pct',0):+.2f}%)

═══ 宏观经济数据 ═══
{econ_str}

请按以下结构输出（用简体中文，不加多余章节）：

【隔夜美股表现】1-2句话总结对A股/港股的传导影响
【宏观与利率环境】1-2句话：Fed政策动态、CPI/非农信号
【VIX与风险偏好】1句话：当前VIX水平对全球风险资产的含义
【地缘政治与关税风险】2-3句话：基于你的知识评估当前中东局势、中美关税动态的潜在影响——明确标注这是定性判断，建议用户核实最新新闻
【对A股影响小结】1-2句话：今日开盘/午后A股可能受到的外部因素影响方向"""


def build_china_policy_prompt() -> str:
    return """请生成今日A股早报的【中国国内政策面与消息面】部分。

基于你的知识，评估近期可能影响A股市场的中国国内因素，包括但不限于：
- 货币政策动态（央行公开市场操作、LPR、存款准备金率等方向性判断）
- 财政政策与产业政策（近期可能的政策导向，如半导体、新能源、医疗等行业支持政策）
- 监管动态（证监会、行业监管相关方向）
- 市场资金面（北向资金、两融余额等，若有相关认知）

请按以下结构输出（用简体中文）：

【货币与财政政策】2-3句话
【产业政策动向】2-3句话，重点关注半导体、新能源、医疗、AI等与持仓相关行业
【监管与市场资金面】1-2句话

⚠️ 重要：必须在末尾明确注明："以上为基于历史知识的定性分析，非实时新闻抓取，请结合今日最新公告及财经媒体核实。" """


def get_macro_commentary(us_snap: dict, us_econ: dict) -> str:
    if not ANTHROPIC_KEY:
        return "[AI未配置] 请检查ANTHROPIC_KEY"
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        prompt = build_macro_section_prompt(us_snap, us_econ)
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=700,
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.error(f"宏观解读生成失败: {e}")
        return f"[AI解读暂时不可用: {e}]"


def get_china_policy_commentary() -> str:
    if not ANTHROPIC_KEY:
        return "[AI未配置]"
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        prompt = build_china_policy_prompt()
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=600,
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.error(f"中国政策解读生成失败: {e}")
        return f"[AI解读暂时不可用: {e}]"


def build_holding_prompt(item: dict, data: dict):
    name, sym, sector = item["name"], item["sym"], item["sector"]
    if not data.get("available"):
        return None

    ma5_str  = data.get("ma5", "N/A")
    ma20_str = data.get("ma20", "N/A")
    rsi_str  = data.get("rsi", "N/A")

    return f"""请用2-3句话简评 {name}（{sym}，{sector}），用简体中文，直接给出操作建议，不要章节标题：

价格: ¥{data['price']} ({data['change_pct']:+.2f}%)
MA5/MA20: {ma5_str} / {ma20_str}
RSI(14): {rsi_str}
成交量比: {data['vol_ratio']:.1f}x

要求：结合技术面给出今日/近期操作倾向（加仓/持有/减仓/观察），一句话即可，不要展开分析过程。"""


def get_holding_commentary(item: dict, data: dict) -> str:
    if not ANTHROPIC_KEY or not data.get("available"):
        return ""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        prompt = build_holding_prompt(item, data)
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=200,
            system="你是A股投资顾问，输出简体中文，直接、简短、无废话。",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.warning(f"{item['sym']} 持仓解读失败: {e}")
        return "（AI解读暂时不可用）"


# ════════════════════════════════════════════════════════════════════════════
#  4. Telegram 发送
# ════════════════════════════════════════════════════════════════════════════

def _send(text: str) -> bool:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.error("Telegram未配置")
        return False
    try:
        resp = requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=15)
        data = resp.json()
        if not data.get("ok"):
            logger.error(f"Telegram错误: {data}")
            return False
        return True
    except requests.RequestException as e:
        logger.error(f"Telegram网络错误: {e}")
        return False


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_header() -> str:
    uk_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"<b>◈ MFTSR ALPHA — 🇨🇳 A股早报</b>\n"
        f"🕐 {uk_now} (UK时间，距A股收盘约1小时)\n"
        f"{'─'*32}"
    )


def build_holding_line(item: dict, data: dict, commentary: str) -> str:
    name, sym = item["name"], item["sym"]
    type_icon = "📦" if item["type"] == "etf" else "📈"

    if not data.get("available"):
        return f"\n{type_icon} <b>{name}</b> ({sym})\n  ⚠️ 数据不可用（yfinance对该代码暂无数据）\n"

    chg = data["change_pct"]
    arrow = "▲" if chg >= 0 else "▼"
    color_emoji = "🟢" if chg >= 1 else ("🔴" if chg <= -1 else "⚪")

    ma_line = ""
    if data.get("above_ma5") is not None and data.get("above_ma20") is not None:
        trend = "多头" if (data["above_ma5"] and data["above_ma20"]) else ("空头" if not (data["above_ma5"] or data["above_ma20"]) else "震荡")
        ma_line = f"  趋势: {trend}  "

    rsi_str = f"RSI={data['rsi']}" if isinstance(data.get("rsi"), (int, float)) else ""

    return (
        f"\n{color_emoji} <b>{name}</b> ({sym})\n"
        f"  💰 ¥{data['price']}  {arrow}{abs(chg):.2f}%  {ma_line}{rsi_str}  Vol={data['vol_ratio']:.1f}x\n"
        f"  {_escape(commentary)}\n"
    )


def send_a_share_report():
    logger.info("开始生成A股早报...")

    # 1. 海外宏观
    us_snap = get_us_overnight_snapshot()
    us_econ = get_us_macro_economic()
    macro_commentary = get_macro_commentary(us_snap, us_econ)
    time.sleep(1)

    # 2. 中国政策面
    china_commentary = get_china_policy_commentary()
    time.sleep(1)

    # 3. 持仓 + 关注清单
    holding_results = []
    for item in A_SHARE_ALL:
        data = get_a_share_data(item["sym"])
        commentary = get_holding_commentary(item, data) if data.get("available") else ""
        holding_results.append({"item": item, "data": data, "commentary": commentary})
        time.sleep(1.5)

    # ── 发送 ──
    header_sent = _send(build_header())
    if not header_sent:
        logger.error(
            "首条消息发送失败，可能是 TELEGRAM_TOKEN 或 CHAT_ID 配置错误。"
            "终止本次报告发送（后续消息大概率也会失败），workflow将标记为失败以便排查。"
        )
        sys.exit(1)
    time.sleep(1)

    _send(f"\n<b>🌐 海外宏观背景</b>\n{'─'*28}\n{_escape(macro_commentary)}")
    time.sleep(1.5)

    _send(f"\n<b>🇨🇳 国内政策面与消息面</b>\n{'─'*28}\n{_escape(china_commentary)}")
    time.sleep(1.5)

    # 持仓
    holdings_msg = f"\n<b>📊 持仓监控</b>\n{'─'*28}\n"
    for r in holding_results:
        if r["item"]["sym"] in [h["sym"] for h in A_SHARE_HOLDINGS]:
            holdings_msg += build_holding_line(r["item"], r["data"], r["commentary"])
    _send(holdings_msg)
    time.sleep(1.5)

    # 关注清单
    watch_msg = f"\n<b>👀 关注清单</b>\n{'─'*28}\n"
    for r in holding_results:
        if r["item"]["sym"] in [w["sym"] for w in A_SHARE_WATCHLIST]:
            watch_msg += build_holding_line(r["item"], r["data"], r["commentary"])
    _send(watch_msg)
    time.sleep(1)

    _send("\n❓ 以上内容仅供参考，不构成投资建议。地缘政治/政策面分析为AI定性判断，请结合最新公告核实。")

    logger.info("A股早报发送完成 ✅")

    try:
        export_a_share_json(us_snap, china_commentary, macro_commentary, holding_results)
    except Exception as e:
        logger.error(f"A股Dashboard JSON导出失败（不影响Telegram报告已发送）: {e}")


def export_a_share_json(us_snap, china_commentary, macro_commentary, holding_results, path="docs/a_share_data.json"):
    """
    把A股早报的真实数据导出为JSON，供Dashboard网页读取。
    输出路径与美股的 dashboard_data.json 分开（a_share_data.json），
    前端会分别fetch两个文件，在Dashboard里用"美股/A股"切换显示。
    """
    import json

    def safe(v):
        if v is None:
            return None
        if isinstance(v, (int, float, str, bool)):
            return v
        return str(v)

    payload = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at_uk": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "macro_commentary": macro_commentary,
        "china_commentary": china_commentary,
        "us_overnight": {
            "sp500_chg_pct": safe(us_snap.get("sp500_chg_pct")),
            "nasdaq_chg_pct": safe(us_snap.get("nasdaq_chg_pct")),
            "dow_chg_pct": safe(us_snap.get("dow_chg_pct")),
            "vix": safe(us_snap.get("vix")),
            "us10y": safe(us_snap.get("us10y")),
            "dxy": safe(us_snap.get("dxy")),
            "crude": safe(us_snap.get("crude")),
        },
        "holdings": [],
        "watchlist": [],
    }

    holding_syms = {h["sym"] for h in A_SHARE_HOLDINGS}
    watchlist_syms = {w["sym"] for w in A_SHARE_WATCHLIST}

    for r in holding_results:
        item, data, commentary = r["item"], r["data"], r["commentary"]
        entry = {
            "sym": item["sym"], "name": item["name"], "type": item["type"], "sector": item["sector"],
            "available": data.get("available", False),
            "price": safe(data.get("price")), "change_pct": safe(data.get("change_pct")),
            "ma5": safe(data.get("ma5")), "ma20": safe(data.get("ma20")),
            "rsi": safe(data.get("rsi")), "vol_ratio": safe(data.get("vol_ratio")),
            "commentary": commentary,
        }
        if item["sym"] in holding_syms:
            payload["holdings"].append(entry)
        elif item["sym"] in watchlist_syms:
            payload["watchlist"].append(entry)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"A股Dashboard JSON导出完成: {path} ({len(payload['holdings'])}持仓 + {len(payload['watchlist'])}关注)")


def main():
    send_a_share_report()


if __name__ == "__main__":
    main()
