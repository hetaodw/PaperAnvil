<div align="center">
  <h1>🚀 PaperAnvil</h1>
  <p><strong>全自动 AI 学术调研报告与分析生成系统</strong></p>
</div>

## 📖 项目简介

**PaperAnvil** 是一个基于多智能体（Multi-Agent）架构的自动化市场调研与学术分析系统。系统内置多个专精于不同数据科学领域的 AI 智能体，通过 LangGraph 的有向无环图进行精准调度，紧密协作，实现从**最初始的调研主题**到**最终的深度研报/学术论文**的端到端全自动生成。

项目深度集成了多种机器学习与数据挖掘范式（如异常检测、K-Means 聚类、LDA 主题建模、语义向量化等），能够**零人工干预**地完成：
- 结构化问卷与交互性 UI 生成
- 高保真用户画像建模与海量样本数据（模拟）扩增
- 跨维度定量定性数据挖掘与高级特征提取
- 高级直观的数据可视化生成
- 多维度学术研报撰写

它极大降低了市场研究与社会学学术调查的门槛，为个人研究、企业战略分析与学术写作提供高质量的数据洞察与“一键式”底稿生成。

## 🎯 核心架构与 Agent 工作流

整个工作流通过 `app.py` 驱动，并内置了自动报错拦截与并发分支机制：

1. 📝 **Survey Agent (调研问卷设计专家)**
   - 根据输入的主题自动设计问卷的第一部分与第二部分。
   - 自动包含人口统计学特征、李克特量表题及深度开放题。
2. 💻 **Survey UI Agent (前端展示专家)**
   - 读取问卷结构，基于深色主题和拟物化玻璃特效生成现代化的问卷 HTML 页面。
3. 🧑‍🤝‍🧑 **Persona Agent (用户画像专家)**
   - 构建多维度、精细化的被试群体“数字孪生”画像及基准分布情况。
4. 🗣️ **Respondent Agent (角色扮演答题专家)**
   - 深入画像视角沉浸式“试玩”问卷，产出种子回答数据。
5. 🔀 **[并行分支: 数据扩充与深度开放题]**
   - **Data Expansion Agent**：基于基准分布混入统计学扰动与噪声，批量生产 5000 条仿真日志数据。
   - **Open-Ended Agent**：精炼定性分析资源，通过 LLM 深度生成不同角色针对开放主观题的情绪性真实反馈。
6. 🔬 **Analysis Agent (数据科学专家)**
   - 利用 `scikit-learn` 与文本语义聚类处理数值型特征与开放型主观反馈。提取核心洞察与可视化指示。
7. 📊 **Plotting Agent (数据可视化专家)**
   - 利用多模态绘图模型生成直观的高级数据信息图表。
8. ✍️ **Writer Agent (学术写作专家)**
   - 合并研究背景、分析过程、数据结果、图示与结论，最终定稿专业的 Markdown 学术论文。

## 🛠️ 环境依赖与设置

### 1. 基础环境
- **Python_**: >= 3.8
- `pip`

### 2. 安装依赖包
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
项目根目录复制 `.env` 样板并填入大语言模型的 API Keys（本项目核心推荐并默认使用 **阿里云百炼 DashScope 的 Qwen** 系列模型）：
```bash
cp .env.example .env
```
示例 `.env`：
```env
# 建议配置 DashScope 提高图文一致性
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
```

## 🚀 运行与自定义配置

### 一键运行
系统通过入口文件 `app.py` 将所有流程串联。只需在同级目录下执行：

```bash
python app.py
```
运行过程中会在终端（同时备份至 `data/output/run.log`）实时打印出每一步的决策、LLM 输出与子工具执行进展。

### ⚙️ 高级配置：自定义 Agent 提示词

为提供最大化的扩展性，**所有核心智能体使用的 System Prompts 已全部抽取至 `app.py` 中集中管理**：

- 在 `app.py` 文件顶部定义了 `DEFAULT_PROMPTS` 字典。
- 用户可直接在 `app.py` 中免改源码地调整任意一个环节的 Prompt (例如调整学术论文的行文风格、问卷设计的侧重点、绘制图表的美学偏好等)。
- 调整完毕后，这些提示词模板会随 `SystemState` 流转并自动注入各节点。

## 📂 项目结构层级

```text
PaperAnvil/
├── app.py                # 🚀 核心工作流入口与提示词统一配置中心
├── requirements.txt      # 环境依赖
├── .env                  # 环境变量配置
├── README.md             # 本说明文档
├── data/                 # 数据沙箱存放区（内含 raw_data, intermediate, processed, output）
├── src/
│   ├── agents/           # 🤖 分工明确的 8 大智能体
│   ├── tools/            # 🔧 纯代码实现的基础原子化工具（统计、语义聚类等）
│   └── workflow/         # 🧠 状态管理 (state.py) 与流向控制
└── tests/                # 单元侧分节点测试与流测试验证集
```

## 📄 测试与二次开发
- 您可以使用 `pytest` 跑通项目里的集成与单元测试，用于调试分支逻辑与并行执行。
- 更多详尽的开发者协作规范可通过 `docs/` 文件夹拓展了解。

---
<div align="center">
  <p><i>Building automated insights from noise, forging papers on the anvil of Multi-Agent AI.</i></p>
  <p>Released under the MIT License.</p>
</div>
