# FxxKStock

一个强调多源信息、事实校验和结构化反偏见的中文证券研究系统。

FxxKStock 面向 A 股、港股、中国 ADR 及海外证券研究场景，将行情、公告、
新闻、社区讨论和宏观数据组织成可追踪的证据，再通过独立盲评、交叉质询、
证伪审计和历史预测校准形成完整研究报告。项目提供实时 Web 工作台、按标的
持久化记忆和本地 Chrome 数据采集，重点降低单一信息源、事实幻觉与模型偏见
对投资结论的影响。

> 本项目用于技术研究和辅助分析，不构成投资建议。模型输出、网络数据和历史表现均可能存在错误，请勿将结果直接用于真实交易决策。

## 为什么是 FxxKStock

证券分析首先是信息质量问题，其次才是模型能力问题。本项目不把“让更多 Agent
讨论”本身当作可靠性的保证，而是重点处理三个更基础的问题：

1. **信息是否足够多元**：行情、公告、新闻、社区讨论和宏观数据来自不同类型、
   不同立场的数据源，避免将单一平台的叙事误当成市场共识。
2. **关键事实是否经过校验**：标的身份、交易日期、OHLCV、当前价格和来源链接
   由程序确定性验证；模型不能仅凭上下文猜测当前价格或公司身份。
3. **结论是否经得起反证**：证据账本区分观察、计算、推断和观点；Bull/Bear
   先独立盲评，再交叉质询；最终计划还要经过独立证伪审计。

### 信息从哪里来

| 信息维度 | 主要来源 | 在研究中的作用 |
|---|---|---|
| 行情与成交 | Yahoo Finance、国内验证快照 | OHLCV、当前价格、成交量和技术指标 |
| 法定披露 | 巨潮资讯 CNINFO | 公告、财报、股东及公司事件 |
| 中文财经新闻 | 东方财富、国内新闻页面 | 公司事件、行业动态和宏观叙事 |
| 投资者社区 | 东方财富股吧、雪球、NGA 大时代 | 散户关注点、争议和情绪，不作为事实源 |
| 海外社区 | StockTwits、Reddit | 海外标的情绪和市场讨论 |
| 宏观数据 | FRED、中文宏观新闻 | 利率、通胀及宏观环境 |
| 预测市场 | Polymarket | 市场隐含概率，仅作为辅助信号 |
| 历史结果 | 本地 ticker memory、预测校准记录 | 对照上次结论、实际收益和预测命中情况 |

系统不会把这些来源简单拼接。官方公告、市场数据、媒体报道和社区观点拥有不同的
证据等级；同源转载不会被当作多个独立来源。

### 如何提高正确性

FxxKStock 不承诺“保证正确”，而是通过可检查的机制降低错误概率：

- **标的身份锁定**：分析开始时解析代码、交易所、证券类型、官方名称和报价币种，
  同一轮所有 Agent 共用该身份。
- **权威行情快照**：当前价和日线 OHLC 分离处理，价格相关结论必须与验证快照一致；
  冲突时要求 Portfolio Manager 重写，仍冲突则拒绝结果。
- **时间边界控制**：行情、新闻和历史报告均带数据日期，避免未来信息泄漏及把旧价格
  描述成现价。
- **来源可追踪**：新闻和公告尽量保留原始链接；无 URL 的数值必须标注工具、日期和
  指标，不允许虚构链接。
- **结构化证据账本**：最多保留 20 条决定性证据，每条记录来源、日期、独立来源数、
  方向、置信度、反证和支持状态。
- **失败显式降级**：数据源失败、名称缺失或结构化解析失败时标记 unavailable/pending，
  继续主流程但降低置信度，不用模型补造缺失事实。
- **历史预测校准**：保存 5/20 交易日评级与价格预测，后续运行自动计算收益、Alpha
  和命中情况，让高置信度结论接受真实结果检验。

### 如何保持观点多元

多元性不仅是“接入更多网站”，还包括对来源相关性和分析角色的约束：

- 官方披露、行情、媒体与社区分别进入不同分析维度。
- 证据账本去重同源转载，并标记单一来源和冲突证据。
- 可研究性评估单独检查来源多样性、共识风险和关键数据缺口。
- Blind Bull 与 Blind Bear 读取相同材料但互相看不到结论，减少锚定和从众。
- 独立盲评完成后才进入多轮交叉质询，研究经理区分“独立共同点”和“辩论后共识”。
- 证伪审计主动寻找最强反方、隐含假设和确认偏误；严重问题只允许打回修正一次，
  避免无限循环产生虚假确定性。
- 最终分别输出数据、论点和执行置信度，禁止用某一维度的高置信度替代其他维度。

## 项目定位

传统的单轮大模型分析容易出现上下文割裂、事实缺少验证、重复抓取数据和结论不可追踪等问题。本项目将证券研究拆分为多个职责明确的智能体，让它们围绕同一份市场数据分别分析、相互辩论，并由风险团队和投资组合经理形成最终意见。

当前版本重点解决以下问题：

- 面向 A 股、港股和中国 ADR 自动切换中文数据源。
- 对关键行情数据进行确定性校验，减少价格和标的身份幻觉。
- 为每只股票保存独立记忆，后续分析在历史结论上增量更新。
- 提供实时 Web 工作台，展示 Agent 执行阶段、工具调用和最终报告。
- 自动启动并复用本地 Google Chrome，通过 CDP 获取动态网页数据。
- 支持多家云端模型、本地模型和 OpenAI-compatible 服务。

## 核心能力

| 能力 | 说明 |
|---|---|
| 多智能体协作 | 市场、情绪、新闻、基本面、研究、交易和风险智能体分工协作 |
| 独立盲评与质询 | Bull/Bear 先独立判断，再进行多轮交叉质询 |
| 风险评审 | 激进、中性、保守三类风险角色审查交易计划 |
| 结构化反偏见 | 证据账本、可研究性评级、独立证伪审计、单次修正及三维置信度 |
| 中国市场适配 | 支持 `.SS`、`.SZ`、`.HK` 及常见中国 ADR |
| 中文数据源 | 东方财富、雪球、NGA 大时代、巨潮资讯及中文宏观新闻 |
| 行情事实校验 | 对标的身份、OHLCV、指标日期和数据窗口进行验证 |
| 股票长期记忆 | 保存每只股票的历史报告、最终决策和事后反思 |
| 历史预测校准 | 自动解析 5/20 交易日结果，统计收益、Alpha 和预测命中率 |
| 增量分析 | 默认复用 30 天内的基本面，刷新行情、新闻和情绪 |
| Web 工作台 | 历史报告、实时阶段、运行状态、模型配置和记忆状态 |
| Chrome 自动管理 | Windows、Ubuntu、macOS 自动启动 Google Chrome CDP |
| 多模型支持 | OpenAI、Gemini、Claude、DeepSeek、Qwen、GLM、MiniMax、Ollama 等 |
| 故障恢复 | 可选 LangGraph SQLite checkpoint，支持中断后恢复 |

## 智能体工作流程

```mermaid
flowchart TD
    INPUT["股票代码与分析日期"] --> ID["标的身份与市场识别"]
    ID --> DATA["数据获取与事实校验"]

    subgraph ANALYSTS["第一阶段：多源分析"]
        direction TB
        MARKET["市场分析师<br/>价格、趋势、技术指标"]
        SENTIMENT["情绪分析师<br/>社区与市场情绪"]
        NEWS["新闻分析师<br/>公司、行业与宏观事件"]
        FUND["基本面分析师<br/>财务、估值与质量"]
    end

    DATA --> MARKET
    DATA --> SENTIMENT
    DATA --> NEWS
    DATA --> FUND

    MARKET --> LEDGER
    SENTIMENT --> LEDGER
    NEWS --> LEDGER
    FUND --> LEDGER

    subgraph RESEARCH["第二阶段：证据与反偏见研究"]
        direction TB
        LEDGER["结构化证据账本<br/>来源、日期、反证与状态"]
        AUDIT0["可研究性评估<br/>来源多样性与数据缺口"]
        BLIND["Bull / Bear 独立盲评<br/>互相不可见"]
        CROSS["Bull / Bear 交叉质询<br/>按研究深度进行多轮辩论"]
        RM["Research Manager<br/>形成研究计划"]
        FALSIFY["独立证伪审计<br/>检查强反证与隐含假设"]
        REVISION{"是否存在严重问题"}
        RM2["Research Manager<br/>单次修正版"]

        LEDGER --> AUDIT0
        AUDIT0 --> BLIND
        BLIND --> CROSS
        CROSS --> RM
        RM --> FALSIFY
        FALSIFY --> REVISION
        REVISION -->|"需要修正"| RM2
    end

    subgraph EXECUTION["第三阶段：交易与风险评审"]
        direction TB
        TRADER["交易员<br/>生成执行计划"]
        RISK["风险团队交叉评审<br/>激进 / 中性 / 保守"]
        PM["投资组合经理"]
        OUT["最终评级、三维置信度<br/>与可验证预测"]

        TRADER --> RISK
        RISK --> PM
        PM --> OUT
    end

    REVISION -->|"审计通过"| TRADER
    RM2 --> TRADER
    OUT --> CAL["5/20 交易日历史校准"]
```

### 各团队职责

1. **分析师团队**
   - 市场分析师：价格走势、成交量、技术指标和关键价位。
   - 情绪分析师：新闻、StockTwits、Reddit 或中文社区情绪。
   - 新闻分析师：公司新闻、宏观数据、预测市场和重大事件。
   - 基本面分析师：财务报表、盈利质量、资产负债和估值。
2. **研究团队**
   - 证据账本区分事实、计算、推断和观点，并记录冲突证据。
   - 多头和空头研究员基于相同材料独立盲评，之后再交叉质询。
   - 研究经理区分独立共同点与辩论后共识，形成投资研究结论。
   - 独立证伪审计检查强反证、隐含假设和认知偏误。
3. **交易与风险团队**
   - 交易员将研究结论转化为仓位和交易计划。
   - 三类风险分析师从不同风险偏好审查方案。
   - 投资组合经理输出最终 Buy、Overweight、Hold、Underweight 或 Sell，
     并分别给出数据、论点和执行置信度及最多三条可验证预测。

## 按股票持久化记忆

每只股票都有独立记忆文件：

```text
memory/
├── trading_memory.md
├── tickers/
│   ├── 513100.SS.json
│   └── 600353.SS.json
└── calibration/
    ├── 513100.SS.json
    └── 600353.SS.json
```

首次分析执行完整流程。后续分析默认进入增量模式：

```mermaid
flowchart TD
    START["开始分析"] --> LOAD{"是否存在该股票记忆"}
    LOAD -->|否| FULL["完整分析<br/>市场 + 情绪 + 新闻 + 基本面"]
    LOAD -->|是| TTL{"基本面是否超过 30 天"}
    TTL -->|否| INC["增量分析<br/>刷新市场 + 情绪 + 新闻<br/>复用基本面"]
    TTL -->|是| FULL
    FULL --> DEBATE["重新执行多空辩论、交易和风险评审"]
    INC --> DEBATE
    DEBATE --> SAVE["原子更新 ticker JSON<br/>追加决策与反思"]
    SAVE --> NEXT["下次分析继续使用"]
```

记忆内容包括：

- 最近分析日期和累计分析次数。
- 市场、情绪、新闻、基本面四类报告。
- 最近一次最终决策。
- 最近一次反偏见审计、证伪条件和三维置信度。
- 基本面数据日期和更新时间。
- 历史决策的实际收益、相对基准收益和反思。
- 待解析及已到期的 5/20 交易日预测校准记录。

Web 中可选择：

- `Auto Incremental`：自动复用有效记忆。
- `Full Refresh`：强制重新运行全部分析师，但仍向 Agent 提供历史上下文。

`memory/` 属于本地运行数据，默认不会提交到 Git。

## 中国市场数据流程

系统会根据交易所后缀、标的元数据和 ADR 列表识别市场区域。

```mermaid
flowchart TD
    TICKER["输入股票代码"] --> REGION{"市场识别"}
    REGION -->|美股及其他市场| GLOBAL["Yahoo Finance<br/>StockTwits / Reddit<br/>英文新闻"]
    REGION -->|A 股 / 港股 / 中国 ADR| CDP{"Chrome CDP 是否可用"}
    CDP -->|是| BROWSER["真实 Chrome<br/>东方财富 / 雪球 / NGA 大时代"]
    CDP -->|否| START["自动启动 Chrome<br/>Windows / Ubuntu / macOS"]
    START --> READY{"15 秒内就绪"}
    READY -->|是| BROWSER
    READY -->|否| HTTP["HTTP 数据源降级"]
    BROWSER --> CNINFO["巨潮资讯公告与股东信息"]
    HTTP --> CNINFO
    GLOBAL --> VERIFY["时间窗口与数据校验"]
    CNINFO --> VERIFY
    VERIFY --> AGENTS["分析师团队"]
```

中国市场默认数据策略：

| 数据类型 | 默认/海外市场 | 中国相关标的 |
|---|---|---|
| 行情与技术指标 | Yahoo Finance | Yahoo Finance + 验证快照 |
| 公司新闻 | Yahoo Finance | Chrome/CDP，失败后东方财富 HTTP |
| 宏观新闻 | 英文新闻与 FRED | 中文宏观新闻与国内数据源 |
| 社区情绪 | StockTwits、Reddit | 东方财富股吧、雪球、NGA 大时代 |
| 官方公告 | 供应商数据 | 巨潮资讯 CNINFO |
| 预测市场 | Polymarket | Polymarket，失败时降级 |

## Chrome 自动启动

分析中国市场标的前，系统会检查 `http://127.0.0.1:9222/json/version`。如果没有可用 CDP，会根据用户选择自动启动桌面版 Google Chrome。

支持的平台：

- macOS
- Windows
- Ubuntu

Chrome 使用项目专用配置目录：

```text
browser_data/chrome-profile/
```

第一次启动后，可在该 Chrome 窗口登录需要 Cookie 的网站。Chrome 会保持运行，后续分析直接复用。该目录可能包含登录信息，已加入 `.gitignore`，请勿公开或提交。

中文社区情绪默认同时启用东方财富、雪球和 NGA。NGA 数据来自“大时代”
版块（`fid=706`），系统会用证券中文名称搜索个股主题，并在可识别时补充行业主题，
再读取近期主题帖及回复。NGA 和雪球可能要求登录；可在 Web 设置页的“浏览器登录”
区域打开对应网站并登录，登录状态会保存在上述专用 Chrome Profile 中。

如果 Chrome 启动失败，系统不会终止分析，而是自动回退到 HTTP 数据源。

## Web 工作台

Web 端提供：

- 历史报告搜索和决策标签。
- 实时 Agent 阶段时间线。
- 工具调用摘要和数据视图。
- 模型供应商、Quick/Deep Model 和研究深度配置。
- 股票记忆状态与增量/全量模式。
- Chrome 平台、自动启动和 CDP 状态。
- 最终 Markdown 报告查看。

### 启动

```bash
conda activate fxxkstock
cd /path/to/FxxKStock

pip install -e ".[webapp]"
python -m webapp.server
```

访问：

```text
http://localhost:8000
```

也可以使用安装后的命令：

```bash
fxxkstock-web
```

## 快速开始

### 1. 环境要求

- Python 3.10 及以上，推荐 Python 3.12。
- Google Chrome，仅中国市场浏览器数据源需要。
- 至少一个可用的 LLM API Key，或本地 Ollama/OpenAI-compatible 服务。

### 2. 创建环境

```bash
git clone <你的仓库地址>
cd FxxKStock

conda create -n fxxkstock python=3.12
conda activate fxxkstock
pip install -e ".[webapp]"
```

开发和测试环境：

```bash
pip install -e ".[dev,webapp]"
```

### 3. 配置模型

```bash
cp .env.example .env
```

在 `.env` 中填写所使用供应商的 Key，例如：

```bash
DEEPSEEK_API_KEY=your-key
FXXKSTOCK_LLM_PROVIDER=deepseek
FXXKSTOCK_QUICK_THINK_LLM=deepseek-v4-flash
FXXKSTOCK_DEEP_THINK_LLM=deepseek-v4-pro
FXXKSTOCK_OUTPUT_LANGUAGE=Chinese
```

支持的主要环境变量：

| 环境变量 | 用途 |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `GOOGLE_API_KEY` | Gemini |
| `ANTHROPIC_API_KEY` | Claude |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `DASHSCOPE_API_KEY` / `DASHSCOPE_CN_API_KEY` | Qwen |
| `ZHIPU_API_KEY` / `ZHIPU_CN_API_KEY` | GLM |
| `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY` | MiniMax |
| `OPENROUTER_API_KEY` | OpenRouter |
| `FRED_API_KEY` | FRED 宏观数据 |
| `FXXKSTOCK_LLM_PROVIDER` | 默认模型供应商 |
| `FXXKSTOCK_LLM_BACKEND_URL` | 自定义兼容接口地址 |
| `FXXKSTOCK_OUTPUT_LANGUAGE` | 报告语言 |
| `FXXKSTOCK_CHROME_PLATFORM` | `macos`、`windows` 或 `ubuntu` |
| `FXXKSTOCK_CHROME_AUTO_START` | 是否自动启动 Chrome |
| `FXXKSTOCK_CHROME_EXECUTABLE` | 自定义 Chrome 可执行文件 |
| `FXXKSTOCK_CHROME_PROFILE_DIR` | 自定义 Chrome Profile 路径 |

### 4. 启动 CLI

```bash
fxxkstock
```

或直接从源码运行：

```bash
python -m cli.main
```

CLI 会依次询问股票代码、分析日期、Chrome 平台、报告语言、分析师、研究深度、模型供应商和模型。

### 5. 直接运行

```bash
python main.py 600519.SS 2026-06-28
```

报告默认保存在：

```text
reports/<TICKER>_<YYYYMMDD_HHMMSS>/
```

## Python 调用

```python
from fxxkstock.default_config import DEFAULT_CONFIG
from fxxkstock.graph.fxxkstock_graph import FxxKStockGraph

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "deepseek"
config["quick_think_llm"] = "deepseek-v4-flash"
config["deep_think_llm"] = "deepseek-v4-pro"
config["output_language"] = "Chinese"
config["cn_browser_platform"] = "macos"

graph = FxxKStockGraph(config=config, debug=True)

final_state, decision = graph.propagate(
    "600519.SS",
    "2026-06-28",
    analysis_mode="auto",
)

print(decision)
```

强制完整刷新：

```python
final_state, decision = graph.propagate(
    "600519.SS",
    "2026-06-28",
    analysis_mode="full",
)
```

## 数据持久化与恢复

| 目录 | 内容 |
|---|---|
| `reports/` | 每次运行生成的完整 Markdown 报告 |
| `memory/tickers/` | 每只股票的最新结构化记忆 |
| `memory/trading_memory.md` | 决策、实际收益和反思日志 |
| `browser_data/chrome-profile/` | 自动启动 Chrome 的专用配置 |
| `~/.fxxkstock/cache/checkpoints/` | 可选的 LangGraph 中断恢复数据 |

启用 checkpoint：

```bash
fxxkstock analyze --checkpoint
```

程序调用：

```python
config["checkpoint_enabled"] = True
```

checkpoint 用于未完成任务的崩溃恢复；ticker memory 用于不同运行之间的长期分析记忆，两者用途不同。

## 项目结构

```text
FxxKStock/
├── cli/                         # 终端交互界面
├── webapp/                      # FastAPI 服务与 Web 工作台
├── fxxkstock/
│   ├── agents/                  # 分析、研究、交易和风险智能体
│   ├── dataflows/               # 行情、新闻、社区和浏览器数据源
│   ├── graph/                   # LangGraph 工作流与运行生命周期
│   ├── llm_clients/             # 模型供应商适配
│   ├── default_config.py        # 默认配置
│   └── reporting.py             # 报告输出
├── tests/                       # 单元与集成测试
├── reports/                     # 本地报告，不提交 Git
├── memory/                      # 本地股票记忆，不提交 Git
└── browser_data/                # 本地 Chrome Profile，不提交 Git
```

## 测试

安装开发依赖：

```bash
pip install -e ".[dev,webapp]"
```

运行全部测试：

```bash
pytest -q
```

运行关键模块：

```bash
pytest -q tests/test_webapp.py
pytest -q tests/test_ticker_memory.py
pytest -q tests/test_chrome_manager.py
pytest -q tests/test_playwright_web.py
```

涉及真实模型、网络数据或 Chrome 的测试可能需要 API Key、网络和本地浏览器环境。

### 数据源诊断

不运行智能体，仅检查 Chrome CDP、东方财富股吧、雪球、NGA 大时代、
东方财富新闻和 Polymarket：

```bash
python scripts/diagnose_data_sources.py --ticker 159516.SZ --platform macos
```

浏览器加载成功的原始 HTML 会保存到
`logs/source_diagnostics/<timestamp>/`，便于检查页面结构和调整解析器。
东方财富宏观关键词会同时测试浏览器与 HTTP 两条路径。

只诊断 NGA（建议首次使用时先在专用 Chrome 中登录）：

```bash
python scripts/diagnose_data_sources.py \
  --ticker 300308.SZ \
  --platform macos \
  --only-nga
```

如证券名称无法自动解析，或需要指定行业关键词：

```bash
python scripts/diagnose_data_sources.py \
  --ticker 300308.SZ \
  --platform macos \
  --only-nga \
  --nga-query 中际旭创 \
  --nga-industry 光模块 \
  --keep-nga-open
```

NGA 诊断结果会额外保存搜索页、抽样主题页和结构化回复到
`logs/source_diagnostics/<timestamp>/`。`--keep-nga-open` 会保留相关标签页，
方便人工核对搜索结果。

只测试 HTTP 数据源：

```bash
python scripts/diagnose_data_sources.py --ticker 159516.SZ --no-browser
```

输出 JSON：

```bash
python scripts/diagnose_data_sources.py --ticker 159516.SZ --json
```

## 当前阶段

这是在原始 TradingAgents 基础上进行大规模改造后的 FxxKStock 初版，当前目标是建立一套可实际使用、可持续积累股票研究记忆的中文多智能体分析工作台。

已经完成的主要改造：

- 中国市场数据路由，以及东方财富、雪球、NGA 大时代中文社区数据源。
- 标的身份、行情窗口和关键价格校验。
- 证据账本、独立盲评、交叉质询和证伪审计。
- 结构化模型输出、五级投资评级和三维置信度。
- 5/20 交易日评级与价格预测校准。
- Web 实时工作台与历史报告。
- 每只股票独立的持久化记忆和增量分析。
- Windows、Ubuntu、macOS Chrome 自动启动。
- 多供应商模型注册和配置。
- FRED、Polymarket、CNINFO 等扩展数据源。

当前仍建议重点验证：

- 不同市场和数据供应商的长期稳定性。
- 长时间运行和并发任务下的资源管理。
- 各模型结构化输出的一致性。
- 记忆压缩、历史版本和回测评估。
- Windows、Ubuntu、macOS 的真实 Chrome 集成。

## 与上游项目的关系

本项目是 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 的衍生版本，不代表上游官方版本。

上游项目提供了多智能体金融研究的核心思想、LangGraph 工作流基础和原始实现。本仓库在此基础上进行了面向个人使用场景的大范围修改。使用或分发本项目时，应继续遵守仓库中的 Apache License 2.0，并保留原项目的版权和归属信息。

原始论文：

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
  title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
  author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
  year={2025},
  eprint={2412.20138},
  archivePrefix={arXiv},
  primaryClass={q-fin.TR},
  url={https://arxiv.org/abs/2412.20138}
}
```

## License

本项目沿用 Apache License 2.0。完整条款见 [LICENSE](LICENSE)。
