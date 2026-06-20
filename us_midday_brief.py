"""
us_midday_brief.py — 美股简报 · UK时间 16:00 推送

内容结构：
  1. 市场总览：VIX变化、主要指数涨跌幅（道指/标普/纳指/罗素2000）
  2. 今日最强/最弱板块（11个SPDR行业ETF涨跌幅排名代理）
  3. 主题动态（AI/半导体重要新闻、当天重要财报、中东/中美重要事件）—
     每次独立搜索总结最新动态，不与前一晚21:30报告做内容对比
  4. 个股清单：实时价格、涨跌、技术面简析（RSI/MA/量比）、新闻摘要、操作倾向
     （技术面用规则计算，不是完整MFTSR五维评分）

中文输出。

环境变量：
    ANTHROPIC_KEY, TELEGRAM_TOKEN, CHAT_ID — 必需
"""

import os
import time
import logging
from datetime import datetime

import requests
import numpy as np
import yfinance as yf
import anthropic

from shared_config import US_WATCHLIST, US_MAJOR_INDICES, US_SECTOR_ETFS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("midday_brief")

ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ════════════════════════════════════════════════════════════════════════════
#  1. 市场总览：VIX、主要指数
# ════════════════════════════════════════════════════════════════════════════

def get_market_overview() -> dict:
    """VIX当前值及变化 + 主要指数涨跌幅。"""
    result = {"indices": {}}
    try:
        h = yf.Ticker("^VIX").history(period="2d")
        if not h.empty:
            vix = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2]) if len(h) > 1 else vix
            result["vix"] = round(vix, 2)
            result["vix_chg"] = round(vix - prev, 2)
            result["vix_chg_pct"] = round((vix - prev) / prev * 100, 2) if prev else 0
    except Exception as e:
        logger.warning(f"VIX获取失败: {e}")
        result["vix"] = None

    for name, sym in US_MAJOR_INDICES.items():
        try:
            h = yf.Ticker(sym).history(period="2d")
            if not h.empty:
                latest = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2]) if len(h) > 1 else latest
                chg_pct = (latest - prev) / prev * 100 if prev else 0
                result["indices"][name] = {"price": round(latest, 2), "chg_pct": round(chg_pct, 2)}
        except Exception as e:
            logger.warning(f"{name}({sym}) 获取失败: {e}")

    return result


# ════════════════════════════════════════════════════════════════════════════
#  2. 板块强弱：11个SPDR行业ETF涨跌幅排名
# ════════════════════════════════════════════════════════════════════════════

def get_sector_performance() -> list:
    """返回按当日涨跌幅排序的板块列表。"""
    results = []
    for sym, name in US_SECTOR_ETFS.items():
        try:
            h = yf.Ticker(sym).history(period="2d")
            if h.empty:
                continue
            latest = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2]) if len(h) > 1 else latest
            chg_pct = (latest - prev) / prev * 100 if prev else 0
            results.append({"name": name, "sym": sym, "chg_pct": round(chg_pct, 2)})
        except Exception as e:
            logger.warning(f"{sym} 板块ETF获取失败: {e}")
    results.sort(key=lambda x: x["chg_pct"], reverse=True)
    return results


# ════════════════════════════════════════════════════════════════════════════
#  3. 个股数据 + 简化技术面（规则计算，非完整MFTSR）
# ════════════════════════════════════════════════════════════════════════════

def get_stock_technical(sym: str) -> dict:
    """获取价格+简化技术指标：RSI、MA20/50、量比。"""
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="90d", interval="1d")
        if hist.empty:
            return {"sym": sym, "available": False}

        close, volume = hist["Close"], hist["Volume"]
        n_days = len(close)
        latest = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if n_days > 1 else latest
        change_pct = (latest - prev) / prev * 100 if prev else 0

        ma20 = float(close.rolling(20).mean().iloc[-1]) if n_days >= 20 else None
        ma50 = float(close.rolling(50).mean().iloc[-1]) if n_days >= 50 else None

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if n_days >= 14 and not rs.isna().iloc[-1] else None

        avg_vol = float(volume.rolling(20).mean().iloc[-1]) if n_days >= 20 else float(volume.mean())
        vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

        thin_history = n_days < 20

        return {
            "sym": sym, "available": True, "thin_history": thin_history,
            "price": round(latest, 2), "change_pct": round(change_pct, 2),
            "ma20": round(ma20, 2) if ma20 is not None else "N/A",
            "ma50": round(ma50, 2) if ma50 is not None else "N/A",
            "above_ma20": (latest > ma20) if ma20 is not None else None,
            "above_ma50": (latest > ma50) if ma50 is not None else None,
            "rsi": round(rsi, 1) if rsi is not None else "N/A",
            "vol_ratio": round(vol_ratio, 2),
        }
    except Exception as e:
        logger.warning(f"{sym} 技术面获取失败: {e}")
        return {"sym": sym, "available": False}


def technical_summary_zh(data: dict) -> str:
    """基于规则给出一句话技术面简评（不调用AI，省成本）。"""
    if not data.get("available"):
        return "数据不可用"
    if data.get("thin_history"):
        return "上市时间较短，技术指标参考价值有限"

    parts = []
    rsi = data.get("rsi")
    if isinstance(rsi, (int, float)):
        if rsi > 70:
            parts.append(f"RSI={rsi:.0f}超买")
        elif rsi < 30:
            parts.append(f"RSI={rsi:.0f}超卖")
        else:
            parts.append(f"RSI={rsi:.0f}中性")

    above20, above50 = data.get("above_ma20"), data.get("above_ma50")
    if above20 is not None and above50 is not None:
        if above20 and above50:
            parts.append("站上MA20/50，趋势偏多")
        elif not above20 and not above50:
            parts.append("跌破MA20/50，趋势偏空")
        else:
            parts.append("均线交织，趋势不明")

    vol_ratio = data.get("vol_ratio", 1)
    if vol_ratio > 1.5:
        parts.append(f"放量{vol_ratio:.1f}x")
    elif vol_ratio < 0.5:
        parts.append(f"缩量{vol_ratio:.1f}x")

    return "，".join(parts) if parts else "信号中性"


# ════════════════════════════════════════════════════════════════════════════
#  4. Claude API + web_search — 个股新闻 与 三大主题动态
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_STOCK = """你是美股盘中简报分析师。任务：搜索指定股票的最新新闻，
用2-3句话总结对股价的影响，并给出简短操作倾向（关注/持有/谨慎/无需行动）。
风格：直接、简短、无废话；不确定时说"暂无重大消息"；全部用简体中文输出。
禁止逐字引用搜索结果原文超过15个单词，必须用自己的话转述。"""

SYSTEM_PROMPT_THEME = """你是宏观与行业新闻分析师，任务是搜索并总结特定主题的最新动态。
风格：直接、精炼、信息密度高；如果没有重大新闻就明确说"暂无新进展"，不要编造内容；
全部用简体中文输出。禁止逐字引用搜索结果原文超过15个单词，必须转述。"""


def get_stock_news(ticker: dict, price_data: dict) -> str:
    if not ANTHROPIC_KEY:
        return "[AI未配置]"

    sym, name = ticker["sym"], ticker["name"]
    price_context = (f"当前价格 ${price_data['price']}，今日涨跌 {price_data['change_pct']:+.2f}%"
                      if price_data.get("available") else "（价格数据暂不可用）")

    prompt = f"""请搜索 {name}（股票代码{sym}）今天或最近1-2天的最新新闻，{price_context}。

请用以下格式回复（简体中文，不要标题，直接2-3句话）：
[新闻摘要] 今日/近期相关消息总结
[操作倾向] 关注/持有/谨慎/无需行动 + 一句话理由

如果搜索没有发现重大新闻，直接说"暂无重大消息，价格变动符合大盘走势"。"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=400,
            system=SYSTEM_PROMPT_STOCK, messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )
        text_parts = [block.text for block in msg.content if hasattr(block, "text")]
        result = "\n".join(text_parts).strip()
        return result if result else "暂无重大消息"
    except Exception as e:
        logger.warning(f"{sym} 新闻搜索失败: {e}")
        return "（新闻搜索暂时不可用）"


THEMES = {
    "ai_semi":   "AI与半导体行业最新重要新闻（如英伟达、AMD、台积电、博通等相关重大动态、新产品发布、订单/产能消息）",
    "earnings":  "今天美股市场重要公司财报发布及市场反应（如有，列出具体公司及关键数字；如无重要财报则说明）",
    "geopolitics": "中东局势及中美关系/关税政策最新进展（如有新的制裁、谈判、冲突升级或缓和信号）",
}


def get_theme_update(theme_key: str, theme_query: str) -> str:
    if not ANTHROPIC_KEY:
        return "[AI未配置]"

    prompt = f"""请搜索并总结：{theme_query}

请用2-4句话总结今天的最新动态（简体中文）。如果没有重大新进展，直接说"今日暂无新进展"。
不要罗列多条新闻标题，只总结对市场/相关股票有实际影响的关键信息。"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=500,
            system=SYSTEM_PROMPT_THEME, messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )
        text_parts = [block.text for block in msg.content if hasattr(block, "text")]
        result = "\n".join(text_parts).strip()
        return result if result else "今日暂无新进展"
    except Exception as e:
        logger.warning(f"主题{theme_key}搜索失败: {e}")
        return "（搜索暂时不可用）"


# ════════════════════════════════════════════════════════════════════════════
#  5. Telegram 发送
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
        f"<b>◈ MFTSR ALPHA — 🇺🇸 美股简报</b>\n"
        f"🕐 {uk_now} (UK时间)\n"
        f"<i>轻量简报：技术面+市场概览+新闻，完整五维分析见21:30收盘报告</i>\n"
        f"{'─'*32}"
    )


def build_market_overview_msg(overview: dict, sectors: list) -> str:
    vix = overview.get("vix")
    vix_line = (f"😌 VIX: <b>{vix}</b> ({overview.get('vix_chg_pct',0):+.2f}%, {overview.get('vix_chg',0):+.2f})"
                if vix is not None else "VIX: 数据不可用")

    lines = [f"\n<b>📊 市场总览</b>\n{'─'*28}", f"  {vix_line}\n"]

    lines.append("  <b>主要指数</b>:")
    for name, d in overview.get("indices", {}).items():
        arrow = "▲" if d["chg_pct"] >= 0 else "▼"
        lines.append(f"    {name}: {d['price']} {arrow}{abs(d['chg_pct']):.2f}%")

    if sectors:
        lines.append("\n  <b>🔥 最强板块</b>:")
        for s in sectors[:3]:
            lines.append(f"    {s['name']}({s['sym']}): {s['chg_pct']:+.2f}%")
        lines.append("\n  <b>🧊 最弱板块</b>:")
        for s in sectors[-3:][::-1]:
            lines.append(f"    {s['name']}({s['sym']}): {s['chg_pct']:+.2f}%")

    return "\n".join(lines)


def build_theme_msg(theme_updates: dict) -> str:
    labels = {"ai_semi": "🤖 AI/半导体动态", "earnings": "📑 今日重要财报", "geopolitics": "🌍 中东/中美局势"}
    lines = [f"\n<b>📰 主题动态</b>\n{'─'*28}"]
    for key, label in labels.items():
        lines.append(f"\n<b>{label}</b>\n  {_escape(theme_updates.get(key, '暂无数据'))}")
    return "\n".join(lines)


def build_ticker_line(ticker: dict, price_data: dict, tech_summary: str, news: str) -> str:
    sym, name = ticker["sym"], ticker["name"]
    if not price_data.get("available"):
        return f"\n⚪ <b>{sym}</b>  {_escape(name)}\n  ⚠️ 价格数据暂不可用\n"

    chg = price_data["change_pct"]
    arrow = "▲" if chg >= 0 else "▼"
    color = "🟢" if chg >= 1 else ("🔴" if chg <= -1 else "⚪")

    return (
        f"\n{color} <b>{sym}</b>  {_escape(name)}\n"
        f"  💰 <b>${price_data['price']}</b>  {arrow}{abs(chg):.2f}%\n"
        f"  📐 技术面: {_escape(tech_summary)}\n"
        f"  📰 {_escape(news)}\n"
    )


def send_midday_brief():
    logger.info("开始生成美股简报...")

    # 1. 市场总览
    overview = get_market_overview()
    sectors = get_sector_performance()

    # 2. 三大主题动态
    theme_updates = {}
    for key, query in THEMES.items():
        logger.info(f"  搜索主题: {key}...")
        theme_updates[key] = get_theme_update(key, query)
        time.sleep(2)

    # 3. 个股
    results = []
    for ticker in US_WATCHLIST:
        sym = ticker["sym"]
        logger.info(f"  处理 {sym}...")
        tech_data = get_stock_technical(sym)
        tech_summary = technical_summary_zh(tech_data)
        news = get_stock_news(ticker, tech_data)
        results.append({"ticker": ticker, "tech_data": tech_data, "tech_summary": tech_summary, "news": news})
        time.sleep(2)

    # ── 发送 ──
    _send(build_header())
    time.sleep(1)

    _send(build_market_overview_msg(overview, sectors))
    time.sleep(1.5)

    _send(build_theme_msg(theme_updates))
    time.sleep(1.5)

    _send(f"\n<b>📈 个股清单</b>\n{'─'*28}")
    time.sleep(1)

    for r in results:
        msg = build_ticker_line(r["ticker"], r["tech_data"], r["tech_summary"], r["news"])
        _send(msg)
        time.sleep(1.5)

    _send("\n❓ 以上内容仅供参考，不构成投资建议。新闻摘要由AI联网搜索生成，技术面为规则计算（非完整MFTSR）。")
    logger.info("美股简报发送完成 ✅")


def main():
    send_midday_brief()


if __name__ == "__main__":
    main()
