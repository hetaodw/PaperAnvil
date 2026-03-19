import operator
from typing import TypedDict, Annotated


class SystemState(TypedDict):
    """
    多智能体协作的全局状态字典。
    
    该状态对象在 LangGraph 工作流中流转，承载各个 Agent 之间的信息传递。
    严禁 Agent 之间进行点对点的直接变量传递，所有数据必须通过此状态对象流转。
    """
    
    thread_id: str
    """
    任务的唯一追踪 ID。
    
    格式：<模块名称>_<任务类型>_<时间戳>
    示例：user_center_module_auth_analysis_1710748800
    
    用途：用于 Checkpoint 保存与恢复，支持断点续传。
    产出者：main.py 初始化时分配
    """
    
    topic: str
    """
    当前研究的主题。
    
    示例：用户中心模块认证流程分析
    
    用途：指导所有 Agent 的工作方向和内容生成。
    产出者：main.py 初始化时传入
    """
    
    questionnaire: dict
    """
    由 Survey Agent 生成的问卷结构 JSON。
    
    包含内容：
    - 问题列表（question_id, question_text, question_type, options, required）
    - 逻辑跳转关系（logic_jump）
    
    用途：指导后续的数据生成和分析。
    产出者：Survey Agent
    """
    
    personas: list[dict]
    """
    由 Persona Agent 生成的用户画像与比例 JSON。
    
    包含内容：
    - 人口统计学特征（age_range, gender, education, income_level, occupation）
    - 行为特征（usage_frequency, purchase_behavior, engagement_level）
    - 偏好特征（answer_tendencies, satisfaction_baseline）
    - 比例分布（proportion）
    
    用途：指导数据扩充工具生成模拟数据。
    产出者：Persona Agent
    """
    
    seed_responses: list[dict]
    """
    存储每个画像对问卷的初始真实回答。
    
    包含内容：
    - name_tag: 画像标识
    - responses: 对问卷各题的具体回答（包含人口、量表和开放题）
    
    用途：作为后续数据生成（Data Expansion）的“种子”样本库。
    产出者：Persona Agent
    """
    
    raw_data_path: str
    """
    由 Python 扩增脚本生成的 5000 条 CSV 数据的本地相对路径。
    
    格式：data/raw_data/simulated_data.csv
    
    用途：提供给 Analysis Agent 进行数据分析和挖掘。
    产出者：Data Expansion 工具
    """
    
    plot_image_paths: Annotated[list[str], operator.add]
    """
    生成的图表路径列表。
    
    格式：["data/output/plot1.png", "data/output/plot2.png", ...]
    
    用途：
    - 记录所有生成的可视化图表
    - 传递给 Writer Agent 用于论文插图
    - 使用 operator.add 聚合器实现列表追加，支持多张图表
    
    产出者：Plotting Agent
    """
    
    analysis_insights: dict
    """
    机器学习分析得出的核心结论。
    
    包含内容：
    - metric：指标名称
    - value：数值
    - conclusion：结论
    - anomaly：异常标识
    
    用途：
    - 传递给 Plotting Agent 用于生成可视化
    - 传递给 Writer Agent 用于撰写论文
    
    产出者：Analysis Agent
    """
    
    thesis_draft: str
    """
    最终生成的 Markdown 论文/报告草稿。
    
    格式：标准的 Markdown 格式
    
    用途：最终输出物，保存到 data/output/ 目录。
    产出者：Writer Agent
    """
    
    error_logs: Annotated[list[str], operator.add]
    """
    系统运行时的报错堆栈。
    
    格式：["[时间戳] [Agent名] 错误信息\nTraceback...", ...]
    
    用途：
    - 记录所有 Agent 运行时的异常
    - 使用 operator.add 聚合器保留完整的试错历史
    - 支持错误追踪和调试
    
    产出者：所有 Agent（遇到错误时追加）
    """
    
    current_step: str
    """
    当前执行到的节点名称。
    
    可能的值：
    - start：开始
    - survey_agent：调研问卷设计
    - persona_agent：用户画像定义
    - respondent_agent：画像模拟答题
    - data_expansion：数据扩充
    - analysis_agent：数据分析
    - plotting_agent：数据可视化
    - writer_agent：论文撰写
    - error：错误状态
    
    用途：追踪工作流执行进度，支持断点续传。
    产出者：每个 Agent 执行时更新
    """
