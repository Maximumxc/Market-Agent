"""
shared_config.py — 三份报告共用的配置：股票池、ETF清单、常量。
被 us_close_report.py / a_share_report.py / us_midday_brief.py 共同引用。
"""

# ════════════════════════════════════════════════════════════════════════════
#  美股持仓 + 关注清单（21:30收盘报告 + 16:00简报 共用）
# ════════════════════════════════════════════════════════════════════════════

US_WATCHLIST = [
    {"sym": "VUAG.L", "name": "Vanguard S&P 500 UCITS ETF", "sector": "Index ETF"},
    {"sym": "MU",     "name": "Micron Technology",          "sector": "Memory/Semiconductors"},
    {"sym": "MRVL",   "name": "Marvell Technology",         "sector": "Semiconductors"},
    {"sym": "NVDA",   "name": "NVIDIA Corp",                "sector": "Semiconductors/AI"},
    {"sym": "GOOGL",  "name": "Alphabet Inc",                "sector": "Search/AI"},
    {"sym": "MSFT",   "name": "Microsoft Corp",              "sector": "Cloud/AI"},
    {"sym": "AVGO",   "name": "Broadcom Inc",                "sector": "Semiconductors/AI Infra"},
    {"sym": "NEE",    "name": "NextEra Energy",              "sector": "Utilities/Renewables"},
    {"sym": "GEV",    "name": "GE Vernova",                  "sector": "Power/Grid Equipment"},
    {"sym": "CEG",    "name": "Constellation Energy",        "sector": "Nuclear/Power Generation"},
    {"sym": "META",   "name": "Meta Platforms",              "sector": "Digital Ads/AI"},
    {"sym": "NOW",    "name": "ServiceNow",                  "sector": "Enterprise Software"},
    {"sym": "ADBE",   "name": "Adobe Inc",                   "sector": "Software/Creative Cloud"},
    {"sym": "SPCX",   "name": "SpaceX (Space Exploration Technologies)", "sector": "Aerospace/Satellite/AI"},
    # 注：SPCX于2026年6月12日IPO，价格历史极短，长周期技术指标(MA200/MA50/RSI)
    # 在上市初期会持续标注"数据不足"而非误导性数字。
    # 注：SMCI已按用户指示从股票池中彻底移除（不再被任何报告跟踪）。
]


# ════════════════════════════════════════════════════════════════════════════
#  A股持仓 + 关注清单（07:00 A股早报 Telegram推送专用 —— 不含ETF）
#  代码格式：上交所 .SS 后缀，深交所 .SZ 后缀（yfinance惯例）
# ════════════════════════════════════════════════════════════════════════════

A_SHARE_HOLDINGS = [
    {"sym": "688271.SS", "name": "联影医疗",      "type": "stock", "sector": "医疗影像设备"},
    {"sym": "000166.SZ", "name": "申万宏源",      "type": "stock", "sector": "证券"},
    {"sym": "603986.SS", "name": "兆易创新",      "type": "stock", "sector": "半导体/存储芯片设计"},
    {"sym": "688525.SS", "name": "佰维存储",      "type": "stock", "sector": "存储芯片"},
    {"sym": "301308.SZ", "name": "江波龙",        "type": "stock", "sector": "存储芯片"},
    {"sym": "000543.SZ", "name": "皖能电力",      "type": "stock", "sector": "电力"},
    {"sym": "600276.SS", "name": "恒瑞医药",      "type": "stock", "sector": "创新药/医药制造"},
    {"sym": "603259.SS", "name": "药明康德",      "type": "stock", "sector": "医药研发外包(CRO/CDMO)"},
    {"sym": "601666.SS", "name": "平煤股份",      "type": "stock", "sector": "煤炭开采(焦煤)"},
    {"sym": "601001.SS", "name": "晋控煤业",      "type": "stock", "sector": "煤炭开采"},
]

# ════════════════════════════════════════════════════════════════════════════
#  A股相关ETF —— 已从Telegram早报中移除（按用户指示），仅供 Dashboard 网页
#  展示使用（Dashboard希望保留全部ETF + 新增的4只个股）
# ════════════════════════════════════════════════════════════════════════════

A_SHARE_ETFS = [
    {"sym": "512880.SS", "name": "证券ETF",       "type": "etf",   "sector": "证券板块"},
    {"sym": "516100.SS", "name": "金融科技ETF",   "type": "etf",   "sector": "金融科技"},
    {"sym": "159020.SZ", "name": "养殖ETF易方达",  "type": "etf",   "sector": "农业养殖"},
    {"sym": "588990.SS", "name": "科创芯片ETF",   "type": "etf",   "sector": "科创板半导体"},
    {"sym": "159819.SZ", "name": "人工智能ETF",   "type": "etf",   "sector": "人工智能"},
    {"sym": "159740.SZ", "name": "恒生科技ETF",   "type": "etf",   "sector": "港股科技"},
    {"sym": "159920.SZ", "name": "恒生ETF华夏",   "type": "etf",   "sector": "港股大盘"},
    {"sym": "513500.SS", "name": "标普500ETF博时", "type": "etf",   "sector": "美股大盘"},
]

# Dashboard专用：持仓个股 + ETF 全部一起展示（不发Telegram，只用于网页JSON导出）
A_SHARE_DASHBOARD_ALL = A_SHARE_HOLDINGS + A_SHARE_ETFS

A_SHARE_WATCHLIST = [
    {"sym": "300502.SZ", "name": "新易盛", "type": "stock", "sector": "光模块"},
    {"sym": "002156.SZ", "name": "通富微电", "type": "stock", "sector": "半导体封测"},
    {"sym": "300750.SZ", "name": "宁德时代", "type": "stock", "sector": "动力电池"},
]

A_SHARE_ALL = A_SHARE_HOLDINGS + A_SHARE_WATCHLIST


# ════════════════════════════════════════════════════════════════════════════
#  MFTSR 权重与阈值 — 精确对应 MFTSR_AI_Growth_Model_EN.xlsx 的权重体系
#  （宏观20% / 基本面35% / 技术面20% / 情绪10% / 风险15%）
# ════════════════════════════════════════════════════════════════════════════

WEIGHTS = {"macro": 0.20, "fundamental": 0.35, "technical": 0.20, "sentiment": 0.10, "risk": 0.15}
ALERT_DIM = 30


# ════════════════════════════════════════════════════════════════════════════
#  美股主要指数（16:00简报：今日涨跌幅展示）
# ════════════════════════════════════════════════════════════════════════════

US_MAJOR_INDICES = {
    "道指":     "^DJI",
    "标普500":  "^GSPC",
    "纳指":     "^IXIC",
    "罗素2000": "^RUT",
}


# ════════════════════════════════════════════════════════════════════════════
#  美股11个SPDR行业ETF（16:00简报：今日最强/最弱板块代理指标）
# ════════════════════════════════════════════════════════════════════════════

US_SECTOR_ETFS = {
    "XLK": "科技",
    "XLE": "能源",
    "XLF": "金融",
    "XLV": "医疗保健",
    "XLY": "消费可选",
    "XLP": "消费必需",
    "XLI": "工业",
    "XLB": "原材料",
    "XLU": "公用事业",
    "XLRE": "房地产",
    "XLC": "通讯服务",
}
