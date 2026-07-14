# token_usage

按本地日期汇总多个 AI CLI 工具的 token 用量。仅读取本地 JSONL 会话日志中的用量元数据，**不发送任何数据到外部**，也不打印任何 prompt 或 response 内容。

支持统计：Claude Code、Antigravity、MimoCode、OpenCode、Gemini CLI 与 Codex profile 0–3。

## 用法

```bash
# 统计今天（默认时区 Asia/Shanghai）
./token_usage.py

# 指定日期
./token_usage.py --date 2026-07-14

# 指定时区
./token_usage.py --timezone America/New_York

# 统计有史以来全部用量
./token_usage.py --all

# 机器可读 JSON 输出
./token_usage.py --json
```

## 选项

| 选项 | 说明 |
| --- | --- |
| `--date YYYY-MM-DD` | 统计日期；默认是指定时区的今天 |
| `--timezone TIMEZONE` | IANA 时区名（默认 `Asia/Shanghai`） |
| `--home HOME` | 用户主目录；主要用于测试或自定义日志位置 |
| `--full-scan` | 忽略文件修改时间优化，扫描全部 JSONL（恢复过旧日志时使用） |
| `--all` | 统计有史以来的全部用量（忽略 `--date`，自动开启全量扫描） |
| `--json` | 输出机器可读 JSON |

## 环境变量

- `TOKEN_USAGE_TIMEZONE`：默认时区，等价于 `--timezone`。

## 依赖

纯 Python 3 标准库，无需安装第三方包。需要 Python 3.9+（使用 `zoneinfo`）。
