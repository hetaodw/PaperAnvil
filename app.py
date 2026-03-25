import os
import sys
import json
import logging
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Project imports
from src.workflow.state import SystemState
from src.agents.survey_agent import survey_node
from src.agents.text_to_survey_agent import text_to_survey_node
from src.agents.survey_ui_agent import survey_ui_node
from src.agents.persona_agent import persona_node
from src.agents.respondent_agent import respondent_node
from src.agents.data_expansion_agent import data_expansion_node
from src.agents.open_ended_agent import open_ended_node
from src.agents.analysis_agent import analysis_agent_node
from src.agents.plotting_agent import plotting_agent_node
from src.agents.writer_agent import writer_agent_node
from src.agents.exporter_agent import exporter_node

# 加载环境变量
load_dotenv()

# ========== 日志配置与标准输出拦截 ==========
class TeeStdout:
    """将终端输出同时镜像写入到日志文件中"""
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, 'a', encoding='utf-8')
        self.stdout = sys.__stdout__

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# ========== LangGraph 路由钩子 ==========
def should_continue(state: SystemState) -> str:
    """判断当前状态是否有错误，如果有则中断图流转"""
    if state.get("current_step") == "error":
        return "error"
    if state.get("error_logs"):
        return "error"
    return "continue"

def route_entry(state: SystemState) -> str:
    """根据是否提供文本，选择问卷生成节点"""
    if state.get("bypass_survey_agent") and state.get("input_text"):
        return "text_to_survey_agent"
    return "survey_agent"

def pass_through(state: SystemState) -> dict:
    """空节点，用于 Fork 和 Join 路由锚点"""
    return {}

def create_workflow():
    """创建并编译 LangGraph 工作流 (支持条件中断)"""
    
    # 1. 初始化状态图
    workflow = StateGraph(SystemState)
    
    # 2. 添加所有节点 (增加显式的 fork 和 join 去更好地管控并行)
    workflow.add_node("survey_agent", survey_node)
    workflow.add_node("text_to_survey_agent", text_to_survey_node)
    workflow.add_node("survey_ui_agent", survey_ui_node)
    workflow.add_node("persona_agent", persona_node)
    workflow.add_node("respondent_agent", respondent_node)
    
    workflow.add_node("fork_node", pass_through)
    workflow.add_node("data_expansion_agent", data_expansion_node)
    workflow.add_node("open_ended_agent", open_ended_node)
    workflow.add_node("join_node", pass_through)
    
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("plotting_agent", plotting_agent_node)
    workflow.add_node("writer_agent", writer_agent_node)
    workflow.add_node("exporter_agent", exporter_node)
    
    # 3. 设置边和流程
    workflow.add_conditional_edges(START, route_entry, {"text_to_survey_agent": "text_to_survey_agent", "survey_agent": "survey_agent"})
    workflow.add_conditional_edges("text_to_survey_agent", should_continue, {"continue": "survey_ui_agent", "error": END})
    workflow.add_conditional_edges("survey_agent", should_continue, {"continue": "survey_ui_agent", "error": END})
    workflow.add_conditional_edges("survey_ui_agent", should_continue, {"continue": "persona_agent", "error": END})
    workflow.add_conditional_edges("persona_agent", should_continue, {"continue": "respondent_agent", "error": END})
    
    # 到达分叉口
    workflow.add_conditional_edges("respondent_agent", should_continue, {"continue": "fork_node", "error": END})
    
    # 并行分支 (Fork)
    workflow.add_edge("fork_node", "data_expansion_agent")
    workflow.add_edge("fork_node", "open_ended_agent")
    
    # 汇聚分支 (Join)
    workflow.add_edge("data_expansion_agent", "join_node")
    workflow.add_edge("open_ended_agent", "join_node")
    
    # 汇聚合并后进入分析节点前检查分支是有无报错
    workflow.add_conditional_edges("join_node", should_continue, {"continue": "analysis_agent", "error": END})
    
    # 最终顺序流
    workflow.add_conditional_edges("analysis_agent", should_continue, {"continue": "plotting_agent", "error": END})
    workflow.add_conditional_edges("plotting_agent", should_continue, {"continue": "writer_agent", "error": END})
    workflow.add_conditional_edges("writer_agent", should_continue, {"continue": "exporter_agent", "error": END})
    workflow.add_edge("exporter_agent", END)
    
    # 4. 编译
    return workflow.compile()

DEFAULT_PROMPTS = {
    "text_to_survey_prompt": """你是一位资深的社会学研究员与数据结构工程师。
你的任务是将用户提供的非结构化/半结构化的问卷文本，转换为与系统兼容的标准 JSON 格式。
调研主题：{topic}

用户提供的原始问卷文本如下：
{input_text}

【转换要求】
1. **提取标题**：如果有明确标题，请提取为 `survey_title`；如果没有，请根据主题自动生成一个标题。
2. **分类题目**：
   - 包含在 demographics 中的题：一般为个人基本情况调查题（包含年龄、性别、职业等的单选/多选题），请分配 id: d1, d2... 并提取 options 列表，每一个选项作为一个字符串。
   - 包含在 likert_scales 中的题：量表打分题/满意度矩阵，一般为 1-5 分。请分配 id: l1, l2...并定义 scale_range 为 [1, 5]，提取 labels（如果原始文本没有给出标签，请默认设定 {{"1": "极不满意/极差", "5": "极其满意/极好"}} 或视语境生成）。
   - 包含在 open_ended 中的题：需要手写回答的文字题。请分配 id: o1, o2...。
3. **补充题目（可选）**：如果原始文本中的题目太少，无法支撑正常的调研，请根据主题自行推演并补齐一些相关题目。

只输出 JSON，格式必须如下（绝不要省略任何字段或使用...）：
{{
  "survey_title": "提取或生成的标题",
  "demographics": [{{"id": "d1", "question": "这道题问什么...", "options": ["选项1", "选项2", "..."]}}],
  "likert_scales": [{{"id": "l1", "question": "这道量表题问什么...", "scale_range": [1, 5], "labels": {{"1": "极差", "5": "极好"}}}}],
  "open_ended": [{{"id": "o1", "question": "这道开放题问什么..."}}]
}}""",

    "survey_phase1_prompt": """你是一位资深的社会学研究员。请针对主题“{topic}”设计问卷的第一部分。
要求包含：
1. 问卷标题。
2. 3-5 道人口统计题 (id: d1, d2...)。
3. 前 10 道李克特量表题 (id: l1 到 l10，1-5分，1为极差/极不满意，5为极好/极其满意)。
4. **特别要求**: 问卷必须包含对中国当前在该领域实际情况的考查，以及对可借鉴发展经验的探索性提问。
【注意：必须完整生成题目，绝不能使用省略号 (...)！】

只输出 JSON，格式如下：
{{
  "survey_title": "...",
  "demographics": [{{"id": "d1", "question": "...", "options": [...]}}],
  "likert_scales": [{{"id": "l1", "question": "...", "scale_range": [1, 5], "labels": {{"1": "极差", "5": "极好"}}}}]
}}""",

    "survey_phase2_prompt": """你已经生成了以下 10 道题目：{context_1}。
现在请继续针对主题“{topic}”生成问卷的第二部分。
要求包含：
1. 另外 10 道互不重复的李克特量表题 (id: l11 到 l20，保持 l11, l12... 的顺序)。【注意：必须完整生成 l11 到 l20 共 10 道题的 JSON 对象，绝不能使用省略号 (...)！】
2. 2 道开放性问题 (id: o1, o2)。

只输出 JSON，格式如下：
{{
  "likert_scales": [{{"id": "l11", "question": "...", "scale_range": [1, 5], "labels": {{"1": "极差", "5": "极好"}}}}],
  "open_ended": [{{"id": "o1", "question": "..."}}]
}}""",

    "survey_ui_prompt": """你是一位专业的问卷设计师和排版专家。
你的任务是根据提供的问卷数据结构，生成一个可以直接打印成 PDF 的专业问卷文档（HTML 格式）。

【问卷结构】
{questionnaire_json}

【设计风格要求 - 打印优先！】
1. **打印友好**: 采用 A4 纸张尺寸，适合直接打印，无需切换模式。
2. **配色方案**: 使用黑白或浅色调，确保打印清晰可读。
3. **排版规范**: 
   - 标题居中，字号 18pt，加粗
   - 题目左对齐，字号 12pt
   - 选项使用方框 □ 供勾选
   - 李克特量表使用表格形式，1-5 分横向排列
   - 开放题预留足够的横线供手写
4. **分页控制**: 合理控制每页题目数量，避免题目跨页断裂。
5. **页眉页脚**: 包含问卷标题和页码。

【交互要求】
- 页面顶部提供"打印"按钮，点击直接调用浏览器打印功能。
- 打印时自动隐藏打印按钮。
- 页面加载完成后自动触发打印预览。

【输出要求】
1. 只输出一个完整的 HTML 文件字符串（包含 CSS 和简单的 Vanilla JS）。
2. 不要包含额外的文字说明或 Markdown 标签。
3. 代码要整洁，语义化。
4. CSS 必须包含 @media print 规则，确保打印效果。

直接输出 HTML。""",

    "persona_system_prompt": """你是一位融合了心理学、社会学与大数据建模能力的资深画像架构师。
你的任务是根据调研主题和问卷，设计出极其鲜活、具有'数字孪生'真实感的人物画像。

【调研主题】
{topic}

【画像背景分布要求 - 重要！】
本调研聚焦于"泰安市留学生视角下的中国医疗体系发展"，但调查对象全部为中国本地居民。
- 中国人画像：可分布在北京、上海、山东（如泰安及济南）、广东、四川等全国各地区。

【年龄分布要求 - 必须严格遵守】
- 青年（18-35岁）：约占 33%
- 中年（36-55岁）：约占 33%
- 老年（56岁以上）：约占 33%
- 比例接近 1:1:1，确保三个年龄段均衡分布

【学历分布要求 - 必须严格遵守】
- 高中及以下：约占 35%
- 专科/高职：约占 25%
- 大学本科：约占 25%（降低比例）
- 硕士及以上：约占 15%
- 重点增加低学历群体（高中、专科）的比例，减少本科及以上学历的占比

【问卷结构】
{demographics_info}
{likert_info}
{open_ended_info}{existing_context}

【每个画像必须包含以下字段】
- name_tag: 姓名风格+标签（如：后端开发-王强）
- gender: 性别（男/女）
- age: 具体年龄（整数）
- job: 具体职业（如：中关村大厂程序员、退休教师等）
- personality: 详细性格描述（如：对系统卡顿极度焦虑、追求极致效率、技术保守派等）
- location: 具体的居住城市及区域（如：北京海淀、上海徐汇）
- proportion: 该类人群在 5000 条样本中的原始预计权重（后续会自动归一化）
- demographics_fixed: 必须匹配问卷中 demographics 的 id，并从对应的 options 中选择一个符合该人设的文本
- likert_distribution: 针对问卷中每个李克特量表题 id，设定其得分均值 mu (1.0-5.0) 和标准差 sigma (0.3-0.9)
- open_ended_samples: JSON对象格式，针对问卷中的开放题 id，以该人物的口吻提供 2 条极其写实、带有性格色彩的口语化回答。例如: {{"o1": "回答1", "o2": "回答2"}}

【输出强制要求】
只输出合法的 JSON 对象，不包含 Markdown 代码块。
JSON 结构：{{"personas": [{{画像1}}, {{画像2}}, ... ]}}""",

    "respondent_system_prompt": """你现在是：{name}。
年龄：{age}
职业：{job}
性格：{personality}
居住地：{location}

你的任务：
请完全沉浸在该人设中，诚实地根据你的性格和背景回答这份问卷。

【问卷内容】
{survey_context}

【回答要求】
1. 必须输出合法的 JSON 格式。
2. 李克特量表题 (likert_scales): 针对每个 id，给出 1-5 之间的整数分。
3. 开放性问题 (open_ended): 针对每个 id，以该角色的口吻写一段 30-50 字的写实回答。
4. 人口统计题 (demographics): 如果问卷中有，请根据你的人设选择合适的选项。

【输出格式示例】
{{
  "responses": {{
    "l1": 4,
    "l2": 2,
    "o1": "具体回答...",
    "d1": "选项文本"
  }}
}}

注意：只输出 JSON 对象，不要包含任何 Markdown 格式的包裹符号（如 ```json）。""",

    "open_ended_system_prompt": """你是一位拥有深厚心理学背景的“沉浸式角色扮演专家”。你擅长完全进入给定的人格画像（Persona），并针对调研问卷中的开放性问题，提供极具真实感、细节丰富且带有强烈个人情绪色彩的深度回答。

# Persona Info
- 姓名标签: {name}
- 年龄: {age}
- 职业: {job}
- 性格特点: {personality}
- 居住地: {location}

# Task
请根据提供的人设信息和问卷题目，以该角色的第一人称口吻进行回答。

# Questionnaire (Open-ended only)
{open_questions_json}

# Constraints
1. **字数要求**：每个问题的回答必须在 150 - 250 字之间。严禁空话套话，必须通过细节描写来填充篇幅。
2. **人设一致性**：回答必须严格符合该角色的职业背景、年龄、性格特征及地理位置。
3. **细节注入**：
   - 必须提到一个具体的**使用场景**（如：在信号不佳的地铁、在忙碌的生产车间、在光线刺眼的户外）。
   - 必须提到一个具体的**操作痛点**（如：某个按钮难以点击、某个页面跳转掉帧、某个报错信息看不懂）。
   - 必须表达一种具体的**情绪**（如：焦虑、愤怒、无助、或偶尔的惊喜）。
4. **输出格式**：只输出合法的 JSON 对象，不包含 Markdown 代码块。

# Output Format
{{
  "persona_name": "{name}",
  "responses": {{
    "o1": "深度回答内容...",
    "o2": "深度回答内容..."
  }}
}}""",

    "analysis_filter_prompt": """你是一个资深的数据科学家。
下面是针对调研数据运行多种算法（基础统计、聚类、异常检测、相关性、LDA主题建模、语义聚类）后得出的综合分析结果。
由于数据量较大，其中包含部分常识性结论。

【综合统计分析结果摘要】
{raw_stats_context} # 防止 Token 超限

【你的任务】
请识别出 3-5 个最具“商业价值”和“学术研究价值”的核心发现。
特别关注：
1. 定量数据中的显著相关或异常簇。
2. 定性数据（语义分析）中频率最高或情感最强烈的反馈。
3. 定量与定性结论的交叉验证。

只输出合法的 JSON 列表，格式如下，不要包含 Markdown 包裹符：
[
  {{"finding_id": "f1", "observation": "发现描述", "why_it_matters": "价值说明"}}
]""",

    "analysis_detailed_prompt": """你是一位首席分析师兼可视化架构师。
基于第一阶段筛选出的高价值发现，请提供深度的解读，并规划可视化的具体路径。

【高价值发现列表】
{high_value_findings_str}

【你的任务】
1. 将发现转化为严谨的学术结论 (conclusion)。
2. 为每个发现规划一个直观的图表 (chart_type 支持: heatmap, bar, scatter, boxplot)。
3. 必须输出合法的 JSON 对象，不包含 Markdown 包裹符。

【输出格式】
{{
  "detailed_insights": [
    {{
      "metric": "指标名称",
      "conclusion": "深入解读 (约100字)",
      "anomaly": "是否涉及异常 (是/否，并简述)"
    }}
  ],
  "visualization_plan": [
    {{
      "chart_type": "图表类型",
      "title": "标题",
      "x_axis": "X轴含义",
      "y_axis": "Y轴含义",
      "plot_instruction": "给 Plotting Agent 的具体绘图建议"
    }}
  ]
}}""",

    "plotting_chart_design_prompt": """你是一个专业的数据可视化与信息图表（Infographic）提示词工程师。
这是上游数据分析阶段提供的核心洞察与可视化规划：
{visualization_plan_json}

【你的任务】
请将上述规划转化为用于调用文生图大模型（Qwen-image）的提示词。
要求生成的图表具有高度的可读性、商业感或学术美感。

提示词必须包含：
1. 图表类型描述（如：极简 3D 柱状图、扁平化饼图、专业热力图）。
2. 具体的文字内容：明确指示在图表中出现的标题、数值、标签（注意：如果有内部引号需求，请严格使用单引号 '，绝不能使用未转义的双引号，否则会导致 JSON 解析失败！）。
3. 视觉风格：指示配色方案（如：科技蓝、商务白）、排版重心。

【输出要求】
必须只输出经过严格校验的合法 JSON 对象（不包含 Markdown 包裹符），格式如下：
{{
  "image_prompts": [
    {{
      "image_id": "chart_01", 
      "prompt": "提示词内容..."
    }}
  ],
  "insertion_guide": [
    {{
      "image_id": "chart_01", 
      "description": "对图表的简短描述", 
      "context": "此图表应放置在关于 X 的讨论之后"
    }}
  ]
}}""",

    "writer_system_setup": "你是一位严谨的高级学术研究员和数据分析师。你的任务是逐步撰写一篇关于《{topic}》的高质量学术报告。请保持语言客观、学术、严谨，不需要任何寒暄或解释，直接输出正文内容。",
    
    "writer_prompt_1": """
【任务】
请撰写报告的第一部分：## 一、 问卷设计。

【输入数据】
这是我们使用的问卷结构：
{questionnaire_json}

【写作要求】
1. 介绍问卷的整体维度（如人口统计、核心体验量表、开放性痛点收集）。
2. 分析为什么采用这些特定的题型（例如：采用 5 分李克特量表是为了量化满意度，开放题是为了捕获具体场景等）。
""",
    
    "writer_prompt_2": """
【任务】
请结合上下文，继续撰写报告的第二部分：## 二、 调查结果。

【输入数据】
这是基于大量有效模拟样本得出的基础统计分布数据：
{basic_stats_json}

【写作要求】
1. 概述样本的总体规模（5000份）。
2. 简明扼要地描述人口统计学特征的分布规律（例如：主要用户群体的年龄段、岗位层级分布、使用的移动设备系统占比等）。
3. 只需要客观陈述数据分布，不需要深入解释原因。
""",
    
    "writer_prompt_3": """
【任务】
请结合上下文，继续撰写报告的第三部分：## 三、 分析方式介绍。

【输入数据】
本次研究主要采用了以下机器学习与统计学分析方法：
{selected_methods}

【写作要求】
1. 逐一介绍上述分析方法的原理。
2. 重点论述将这些方法应用于本次“移动端体验调研”的【学术合理性】和【业务价值】（例如：为什么用 K-Means 能更好地给用户分群？为什么用异常检测找极度不满的用户很重要？）。
3. 语言要体现出数据科学的专业性。
""",

    "writer_prompt_4": """
【任务】
请结合上下文，继续撰写报告的最核心部分：## 四、 分析结果。

【输入数据】
1. 深度分析洞察：
{detailed_insights_json}

2. 待插入的图表指南（包含图片路径和图表含义）：
{image_guide_json}

【写作要求】
1. 根据“深度分析洞察”，分点/分段进行详尽的学术论述。解释数据背后的业务现象（如：不同画像群体的痛点差异）。
2. **强制插图指令**：当你的文字论述到某个图表所展示的内容时，**必须**使用 Markdown 语法插入对应的图片。格式为 `![图表描述](对应的本地路径)`。
3. 确保图文并茂，逻辑递进，不要遗漏任何一个高价值的洞察和图表。
""",

    "writer_prompt_5": """
【任务】
请结合上下文与调研主题的核心探讨点，撰写报告的第五部分：## 五、 中国现状与典型案例分析。

【要求】
1. 深入调研并描述中国当前在该领域（{topic}）的实际现状。
2. 结合数据洞察，分析 1-2 个能够体现“中国特色”或“中国模式”的典型案例或现象。
""",

    "writer_prompt_6": """
【任务】
请结合前文所有分析，撰写报告的最后一部分：## 六、 总结与可借鉴经验建议。

【写作要求】
1. 凝练前文所有的核心调研发现。
2. **重点列出**：中国在这一领域（{topic}）有哪些【可被国际借鉴的发展经验】或【具有普适价值的中国方案】。
3. 提出 3-4 条极具针对性、前瞻性的改进或国际推广建议。
"""
}

def run_workflow(topic: str = None, persona_count: int = 21, log_callback=None, use_stdout_redirect=True, custom_prompts: dict = None, input_text: str = "", resume_persona_checkpoint: bool = False, use_existing_csv: bool = False, existing_csv_path: str = ""):
    """
    运行 LangGraph 工作流。
    
    Args:
        topic: 调研主题
        persona_count: 模拟画像数量
        log_callback: 接收日志的函数回调 (例如用于 GUI 更新)
        use_stdout_redirect: 是否重定向全局 stdout (GUI 模式建议 False)
        custom_prompts: 用户自定义的提示词字典
        input_text: 已有问卷文本
        resume_persona_checkpoint: 是否从断点恢复画像生成
        use_existing_csv: 是否使用用户提供的已有 CSV 文件
        existing_csv_path: 用户提供的 CSV 文件路径
    """
    # 只有在需要时才重定向日志输出
    log_path = "data/output/run.log"
    if use_stdout_redirect:
        sys.stdout = TeeStdout(log_path)
        sys.stderr = TeeStdout(log_path)
    
    def log(msg):
        # 如果是字符串则直接打印/回调，否则转字符串
        msg_str = str(msg)
        print(msg_str)
        if log_callback:
            log_callback(msg_str)

    log("\n" + "🚀" * 30)
    log("      PaperAnvil: 全自动 AI 学术调研报告生成系统")
    log("🚀" * 30 + "\n")
    log(f"📄 系统运行日志将同步保存至: {log_path}\n")
    
    # 构建最终 Prompt
    active_prompts = DEFAULT_PROMPTS.copy()
    if custom_prompts:
        log("检测到自定义提示词，正在应用覆盖...")
        active_prompts.update(custom_prompts)
    
    # 初始化状态
    initial_state: SystemState = {
        "topic": topic if topic else "人口老龄化与医疗服务需求研究 - 泰安市留学生视角下的中国医疗体系发展及中国当前的情况和可被借鉴的发展经验",
        "input_text": input_text,
        "bypass_survey_agent": bool(input_text.strip()),
        "persona_count": persona_count,
        "resume_persona_checkpoint": resume_persona_checkpoint,
        "use_existing_csv": use_existing_csv,
        "existing_csv_path": existing_csv_path,
        "questionnaire": {},
        "personas": [],
        "seed_responses": [],
        "raw_data_path": "",
        "open_ended_detailed_responses": [],
        "analysis_insights": {},
        "thesis_draft": "",
        "plot_image_paths": [],
        "image_insertion_guide": [],
        "basic_stats": {},
        "semantic_stats": {},
        "error_logs": [],
        "current_step": "START",
        "prompts": active_prompts
    }
    
    app = create_workflow()
    
    log(">>> 工作流编译成功，开始进入流转阶段 (LangGraph Trace On)")
    
    try:
        final_state = initial_state.copy()
        
        # 使用 stream 替代 invoke，捕获每个节点的动态更新状态
        for event in app.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in event.items():
                if not state_update:
                    continue
                    
                if node_name not in ["fork_node", "join_node"]:
                    log(f"\n[DEBUG] 节点「{node_name}」执行完成。")
                    
                    # 打印一些关键的状态变更信息
                    for key in state_update:
                        if key == "questionnaire" and state_update[key]:
                            log(f"  └─ 🚀 问卷结构已生成: {state_update[key].get('survey_title')}")
                        elif key == "personas":
                            log(f"  └─ 👥 已生成 {len(state_update[key])} 个虚拟画像")
                        elif key == "seed_responses":
                            log(f"  └─ 📥 已收集 {len(state_update[key])} 条初始调研样本")
                        elif key == "raw_data_path" and state_update[key]:
                            log(f"  └─ 📊 数据文件已准备: {state_update[key]}")
                        elif key == "error_logs" and state_update[key]:
                            log(f"  └─ ⚠️ 检测到错误日志: {state_update[key][-1]}")
                
                # 跟踪具体的流转
                if "current_step" in state_update and node_name not in ["fork_node", "join_node"]:
                    log(f"📦 全局追踪状态变更: {final_state.get('current_step', 'START')} -> {state_update['current_step']}")
                    
                # 收集最新的变更拼接到 final_state 中以作全局引用
                for k, v in state_update.items():
                    if k == "error_logs":
                        final_state.setdefault("error_logs", []).extend(v)
                    elif k == "plot_image_paths":
                        final_state.setdefault("plot_image_paths", []).extend(v)
                    else:
                        final_state[k] = v

        if final_state.get("current_step") == "error" or final_state.get("error_logs"):
            print("\n🚨🚨 工作流因严重错误已被中断 🚨🚨")
            if final_state.get("error_logs"):
                print(f"核心错误内容: {final_state['error_logs'][-1]}")
        else:
            print("\n" + "✅" * 30)
            print("      全流程生产完成！")
            print("✅" * 30 + "\n")
            
            if final_state.get("thesis_draft"):
                print(f"🎉 最终调研报告已生成：data/output/thesis_draft.md")
                print(f"📄 报告字符数：{len(final_state['thesis_draft'])}")
            
            if final_state.get("plot_image_paths"):
                print(f"📊 已生成图表数量：{len(final_state['plot_image_paths'])}")
            
            return final_state
                
    except Exception as e:
        print(f"\n❌ 工作流最外层运行发生异常：{e}")
        import traceback
        traceback.print_exc()
        return final_state
    finally:
        # 恢复 stdout
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def main():
    """CLI 入口点"""
    run_workflow()

if __name__ == "__main__":
    main()
