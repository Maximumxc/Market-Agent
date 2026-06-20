MFTSR Alpha — 自动化投资分析报告系统
基于 Claude AI + GitHub Actions 的美股/A股自动化分析报告，每天定时推送到 Telegram。
三个定时任务
时间 (UK)	任务	文件
07:00	A股早报	`a_share_report.py`
16:00	美股简报	`us_midday_brief.py`
21:30	美股收盘完整报告	`us_close_report.py`
周一至周五自动运行，周末跳过。
文件说明
`shared_config.py` — 股票池/ETF清单/阈值等共用配置
`a_share_report.py` — A股持仓+关注清单早报，含海外宏观与中国政策面解读
`us_midday_brief.py` — 美股盘中简报：市场总览、板块强弱、主题新闻、个股技术面
`us_close_report.py` — 美股收盘完整MFTSR五维评分报告（中英双语）
`.github/workflows/` — 三个GitHub Actions定时任务配置
`SETUP_GUIDE.md` — 完整部署与使用说明
配置
仓库 Secrets 需要设置（Settings → Secrets and variables → Actions）：
`ANTHROPIC_KEY`
`TELEGRAM_TOKEN`
`CHAT_ID`
`FRED_API_KEY`（可选，用于CPI/非农等美国官方经济数据）
详细说明见 `SETUP_GUIDE.md`。
---
仅供个人参考，不构成投资建议
