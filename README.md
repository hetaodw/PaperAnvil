<div align="center">
  <img src="logo.png" alt="PaperAnvil Logo" width="120">
  <h1>PaperAnvil</h1>
  <p><strong>全自动 AI 学术调研报告与分析生成系统</strong></p>
  <p>
    <a href="#-核心功能">功能</a> •
    <a href="#-快速开始">快速开始</a> •
    <a href="#-架构设计">架构</a> •
    <a href="#-项目结构">结构</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/LangGraph-Workflow-orange.svg" alt="LangGraph">
  </p>
</div>

---

## 📖 项目简介

**PaperAnvil** 是一个基于多智能体（Multi-Agent）架构的自动化市场调研与学术分析系统。通过 LangGraph 工作流编排多个专精 AI 智能体，实现从**调研主题**到**学术论文**的端到端全自动生成。

### ✨ 核心特性

- 🎯 **一键生成** - 输入主题，自动完成问卷设计、数据模拟、分析可视化、报告撰写
- 🤖 **多智能体协作** - 8 个专业 Agent 分工明确，通过状态机精准调度
- 📊 **高级数据分析** - 集成 K-Means 聚类、异常检测、LDA 主题建模、语义分析等
- 🖥️ **现代化 GUI** - PySide6 深色主题界面，支持实时日志追踪
- 🔄 **断点续传** - 支持从断点恢复，避免重复计算
- 📁 **外部数据支持** - 可导入已有 CSV 数据进行分析

---

## 🚀 快速开始

### 1. 环境要求

- Python >= 3.8
- 推荐使用 Anaconda 环境

### 2. 安装依赖

```bash
git clone https://github.com/hetaodw/PaperAnvil.git
cd PaperAnvil
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入您的 DashScope API Key
# 获取地址: https://dashscope.console.aliyun.com/
```

### 4. 启动程序

**GUI 模式（推荐）：**
```bash
python gui_main.py
```

**命令行模式：**
```bash
python app.py
```

---

## 🎯 核心功能

### 📝 智能问卷设计
- 自动生成人口统计学问题、李克特量表、开放性问题
- 支持导入已有问卷文本
- 生成现代化 HTML 问卷页面

### 👥 用户画像建模
- 基于人口统计学特征构建高保真画像
- 支持年龄、学历等分布比例自定义
- 自动生成画像比例权重

### 📊 数据模拟扩增
- 基于画像分布生成 2000+ 条模拟数据
- 李克特量表正态分布采样
- 自动处理缺失值和异常值

### 🔬 多维数据分析
- **描述性统计** - 频率分布、均值、标准差
- **机器学习** - K-Means 聚类、DBSCAN、孤立森林异常检测
- **相关性分析** - 特征相关系数矩阵
- **语义分析** - LDA 主题建模、情感分析

### 📈 可视化生成
- 自动生成统计图表
- 支持多图表组合
- 高清 PNG 输出

### ✍️ 学术报告撰写
- 自动生成 Markdown 格式学术论文
- 包含研究背景、方法、结果、讨论、结论
- 自动插入图表引用

---

## 🏗️ 架构设计

### Agent 工作流

```
┌─────────────────────────────────────────────────────────────┐
│                      PaperAnvil Workflow                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Survey Agent │ -> │ Survey UI    │ -> │ Persona      │  │
│  │ 问卷设计      │    │ Agent 界面   │    │ Agent 画像   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                 │            │
│                     ┌───────────────────────────┼──────────┐│
│                     ▼                           ▼          ││
│  ┌──────────────────────┐    ┌──────────────────────────┐ ││
│  │ Data Expansion Agent │    │   Open-Ended Agent       │ ││
│  │ 数据扩增 (2000条)     │    │   开放题深度回答          │ ││
│  └──────────────────────┘    └──────────────────────────┘ ││
│                     │                           │          ││
│                     └───────────┬───────────────┘          ││
│                                 ▼                          ││
│                     ┌──────────────────────┐               ││
│                     │   Analysis Agent     │               ││
│                     │   数据分析 & 洞察     │               ││
│                     └──────────────────────┘               ││
│                                 │                          ││
│                     ┌───────────┴───────────┐              ││
│                     ▼                       ▼              ││
│  ┌──────────────────────┐    ┌──────────────────────┐     ││
│  │  Plotting Agent      │    │   Writer Agent       │     ││
│  │  可视化图表生成       │ -> │   学术报告撰写        │     ││
│  └──────────────────────┘    └──────────────────────┘     ││
│                                        │                   ││
│                                        ▼                   ││
│                           ┌──────────────────────┐         ││
│                           │   Markdown 报告      │         ││
│                           │   Excel 数据表       │         ││
│                           └──────────────────────┘         ││
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 组件 | 技术 |
|------|------|
| 工作流编排 | LangGraph |
| LLM 接口 | OpenAI API (兼容 DashScope) |
| GUI 框架 | PySide6 |
| 数据处理 | Pandas, NumPy |
| 机器学习 | scikit-learn |
| 可视化 | Matplotlib |

---

## 📂 项目结构

```
PaperAnvil/
├── app.py                    # 核心工作流入口 & 提示词配置
├── gui_main.py               # GUI 主程序
├── package.py                # PyInstaller 打包脚本
├── requirements.txt          # 依赖列表
├── .env.example              # 环境变量模板
├── logo.png / logo.ico       # 项目图标
│
├── src/
│   ├── agents/               # 🤖 智能体模块
│   │   ├── survey_agent.py       # 问卷设计
│   │   ├── survey_ui_agent.py    # 问卷界面生成
│   │   ├── persona_agent.py      # 用户画像
│   │   ├── data_expansion_agent.py # 数据扩增
│   │   ├── open_ended_agent.py   # 开放题回答
│   │   ├── analysis_agent.py     # 数据分析
│   │   ├── plotting_agent.py     # 可视化
│   │   └── writer_agent.py       # 报告撰写
│   │
│   ├── tools/                # 🔧 工具模块
│   │   ├── basic_stats_tool.py   # 基础统计
│   │   ├── state_tool.py         # ML 分析
│   │   ├── semantic_tool.py      # 语义分析
│   │   ├── data_expansion.py     # 数据生成
│   │   ├── csv_validator.py      # CSV 验证
│   │   └── csv_to_xlsx.py        # 格式转换
│   │
│   └── workflow/             # 🧠 工作流模块
│       ├── state.py              # 状态定义
│       └── graph.py              # 图结构
│
├── data/
│   ├── assets/               # 静态资源
│   ├── intermediate/         # 中间文件 (git ignored)
│   ├── output/               # 输出文件 (git ignored)
│   └── raw_data/             # 原始数据 (git ignored)
│
├── tests/                    # 🧪 测试用例
└── docs/                     # 📚 文档
```

---

## ⚙️ 高级配置

### 自定义提示词

所有 Agent 的提示词模板都在 `app.py` 的 `DEFAULT_PROMPTS` 字典中，可直接修改：

```python
DEFAULT_PROMPTS = {
    "persona_system_prompt": "你的自定义提示词...",
    "analysis_filter_prompt": "...",
    # ...
}
```

### 画像分布配置

在 `persona_system_prompt` 中可调整：
- 年龄分布（青年/中年/老年比例）
- 学历分布（高中/专科/本科/硕士比例）

### 断点续传

- 画像生成支持断点保存，失败后可从断点恢复
- 勾选 GUI 中的"从断点恢复"选项即可

### 使用外部数据

1. 勾选"使用已有数据文件"
2. 选择 CSV 文件
3. 系统会自动验证并处理数据

---

## 📄 输出示例

运行完成后，在 `data/output/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `thesis_draft.md` | 学术报告 (Markdown) |
| `final_survey_results.xlsx` | 调研数据 (Excel) |
| `plot_*.png` | 可视化图表 |
| `run.log` | 运行日志 |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📜 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<div align="center">
  <p><i>Building automated insights from noise, forging papers on the anvil of Multi-Agent AI.</i></p>
  <p>Made with ❤️ by <a href="https://github.com/hetaodw">hetaodw</a></p>
</div>
