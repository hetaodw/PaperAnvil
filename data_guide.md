# PaperAnvil 数据架构手册：中间文件与分析逻辑说明

本项目在运行过程中会生成一系列 JSON 中间文件，用于连接各 Agent 的工作流并存储多维度的分析结果。本手册旨在帮助开发者与研究员理解这些文件的作用及其详细的数据结构。

---

## 一、 中间过程文件 (data/intermediate/)

### 1. 问卷定义 (`questionnaire.json`)
*   **作用**：调研的基石，定义了所有题目、选项及量表范围。
*   **关键结构**：
    *   `demographics`: 人口统计学题目（如性别、年龄）。
    *   `likert_scales`: 1-5 分量表题目。
    *   `open_ended`: 开放式主观题。

### 2. 虚拟画像 (`personas.json`)
*   **作用**：由 `PersonaAgent` 生成的模拟受访者原型。
*   **关键结构**：
    *   `name_tag`: 画像标签。
    *   `likert_distribution`: 该画像在量表题上的预设均值 (`mu`) 和标准差 (`sigma`)，决定了后续数据生成的偏好。

### 3. 本地响应样本 (`seed_responses.json` & `open_ended_responses.json`)
*   **作用**：存储各画像生成的原始回答样本，作为海量数据扩增的“种子”。

---

## 二、 核心分析输出文件结构详解

### 1. 基础统计报告 (`basic_stats.json`)
本文件由 `BasicStatsTool` 生成，提供定量的现状描述。
*   `demographics_distribution`: 各人口学维度的频数 (`counts`) 和百分比 (`percentages`)。
*   `likert_stats`: 针对每个 `l` 编号的题目：
    *   `mean`: 算术平均分。
    *   `std`: 标准差（衡量意见分歧程度）。
    *   `distribution`: 1-5 分各分值的具体投票人数。

### 2. 深度分析结果 (`analysis_results.json`)
由 `StateTool` 生成，包含机器学习算法的洞察。
*   **`clustering` (聚类分析)**:
    *   `kmeans_centers`: 识别出的各人群团块在 20 个维度上的中心坐标。
    *   `kmeans_counts`: 各类群的人数分布。
*   **`anomaly_detection` (异常检测)**:
    *   `anomaly_percentage`: 异常值占比（默认 5%）。
    *   `example_anomalies`: 被识别出的极端或矛盾样本明细。
*   **`correlation_analysis` (相关性)**:
    *   `matrix`: 变量间的相关系数矩阵（-1 到 1）。
*   **`feature_contribution` (核心贡献度)**:
    *   `importances`: 基于随机森林评估的、对“整体满意度”影响最大的前几项指标。

### 3. 语义分析报告 (`semantic_analysis.json`)
由 `SemanticTool` 生成，处理主观文本数据。
*   **`topic_modeling` (主题建模)**:
    *   `interpretation`: LLM 对 LDA 提取出的关键词进行的结构化总结（核心名称 + 描述）。
*   **`absa` (细粒度情感分析)**:
    *   `aspect`: 提到的具体事物（如：自助设备、排队时间）。
    *   `sentiment`: 情感极性（正面/负面/中性）。
*   **`semantic_clustering` (语义聚类)**:
    *   将相似意思的评论聚类，并提取出 `representative_text` (代表性原声)。

---

## 三、 最终合成文件 (data/processed/)

### 1. 分析洞察汇总 (`analysis_insights.json`)
由 `AnalysisAgent` 结合以上所有 JSON 数据，通过 LLM 过滤掉噪声后生成的“最终裁判”。
*   `detailed_insights`: 包含 `metric` (指标)、`conclusion` (结论) 和 `anomaly` (是否异常) 的结构化列表，直接供 `WriterAgent` 编写论文。
*   `visualization_plan`: 绘图指令清单，规定了 `PlottingAgent` 应该画什么图（bar/scatter/heatmap）以及具体的绘图着色建议。
