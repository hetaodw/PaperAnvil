# PaperAnvil

本项目是一个基于多智能体（Multi-Agent）架构的自动化市场调研与分析系统。系统内置多个专精于不同数据科学领域的 AI 智能体，通过紧密协作，实现从原始市场数据到深度行业研报的端到端生成。系统深度集成了多种机器学习与人工智能分析范式（包括但不限于聚类分析、时间序列预测、NLP 情感分析及消费者行为建模），能够自主完成模拟数据生成、高维数据降维清洗、多维度特征提取、动态数据可视化及基于本地知识库（RAG）的专业论文/研报撰写。本项目旨在通过大语言模型驱动的自主规划与自我修正机制，大幅降低市场分析的门槛与成本，为商业决策和学术研究提供高置信度的数据洞察。

## 项目结构

```
PaperAnvil/
├── app.py                # Web 交互界面入口 (未来可使用 Streamlit/Gradio 编写)
├── main.py               # 命令行直接运行 LangGraph 工作流的主入口
├── requirements.txt      # 依赖清单 (LangGraph, Pandas, OpenAI 等)
├── .env                  # 存放 API Keys 的环境变量文件 (切勿提交到 Git)
├── README.md
├── docs/                 # 项目规范与文档
│   ├── ROLES.md          # 记录各 Agent 的职责边界和 System Prompt 设计
│   ├── DEVELOPMENT_RULES.md # 开发与状态流转规范
│   └── rule.md           # 原始规则文档
├── data/
│   ├── intermediate/     # 存放仿真过程的中间件 (如生成的问卷结构 JSON、画像比例 JSON)
│   ├── raw_data/         # 存放 Python 脚本基于画像最终生成的 5000 条 CSV 模拟日志数据
│   ├── processed/        # 存放 Analysis Agent 清洗或降维后的分析用中间表
│   └── output/           # 存放最终生成的并发图表 (.png) 和 用户中心模块分析论文 (.md)
├── knowledge_base/       # RAG 本地知识库
│   ├── documents/        # 存放系统架构设计文档、用户中心设计白皮书等 PDF/Markdown
│   └── vector_store/     # ChromaDB/FAISS 向量数据库的本地持久化存储
└── src/
    ├── agents/           # 大模型驱动的智能体节点
    │   ├── survey_agent.py      # 节点 1：生成调研问卷结构
    │   ├── persona_agent.py     # 节点 2：定义用户画像与答题基准分布
    │   ├── analysis_agent.py    # 节点 4：执行机器学习聚类与异常检测
    │   ├── plotting_agent.py    # 节点 5：根据结论绘制并发延迟分布图表
    │   └── writer_agent.py      # 节点 6：结合 RAG 撰写学术章节
    ├── tools/            # 纯代码实现的原子化工具与处理节点
    │   ├── python_repl.py       # 核心底层：安全执行 Python 代码的沙箱
    │   ├── data_expansion.py    # 节点 3：【纯代码】读取画像比例，注入噪声并批量扩增数据的脚本
    │   └── rag_retriever.py     # 从 vector_store 检索相关文献的工具
    └── workflow/         # LangGraph 核心调度控制枢纽
        ├── state.py             # 定义全局字典 SystemState (包含 questionnaire, personas 等字段)
        └── graph.py             # 组装节点、定义边 (Edge) 以及错误重试循环
```

## 核心功能模块

### 1. Survey Agent (调研问卷设计专家)
- 根据调研主题生成结构化问卷
- 定义问题类型和逻辑跳转关系
- 输出 JSON 格式的问卷结构

### 2. Persona Agent (用户画像专家)
- 定义目标用户群体和画像
- 设定各画像群体的答题基准分布
- 输出 JSON 格式的用户画像

### 3. Data Expansion (数据扩充工具)
- 读取用户画像比例
- 注入噪声并批量扩增数据
- 生成 5000 条模拟数据

### 4. Analysis Agent (数据科学专家)
- 执行数据清洗和预处理
- 进行机器学习聚类分析
- 检测异常值和异常模式
- 提取关键指标和洞察

### 5. Plotting Agent (数据可视化专家)
- 根据分析结论绘制图表
- 支持多种图表类型
- 生成高质量的可视化输出

### 6. Writer Agent (学术写作专家)
- 结合分析结论和相关文献撰写论文
- 使用 RAG 检索相关文献
- 输出 Markdown 格式的学术报告

## 快速开始

### 环境要求
- Python 3.8+
- pip

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置环境变量
复制 `.env` 文件并填入你的 API Keys：
```bash
cp .env.example .env
```

编辑 `.env` 文件：
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

### 运行项目
```bash
python main.py
```

## 工作流程

1. **Survey Agent** → 生成问卷结构
2. **Persona Agent** → 定义用户画像和答题分布
3. **Data Expansion** → 扩充数据生成 5000 条模拟数据
4. **Analysis Agent** → 执行数据分析和异常检测
5. **Plotting Agent** → 绘制数据可视化图表
6. **Writer Agent** → 撰写最终论文/报告

## 开发规范

请参阅以下文档了解详细的开发规范：
- [ROLES.md](docs/ROLES.md) - Agent 角色定义与职责边界
- [DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md) - 开发与状态流转规范

## 许可证

MIT License
