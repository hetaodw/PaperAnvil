import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.workflow.state import SystemState

load_dotenv()

def writer_agent_node(state: SystemState) -> dict:
    """
    学术写作智能体：通过 5 阶段提示词工程，生成高质量数据调研报告。
    """
    print("\n" + "="*60)
    print("📝 [Writer Agent] 开始分阶段撰写学术调研报告...")
    print("="*60)

    # 初始化 LLM 客户端
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY")),
        base_url=os.getenv("DASHSCOPE_BASE_URL", os.getenv("OPENAI_BASE_URL")),
    )
    model_name = os.getenv("MODEL_NAME", "gpt-4o")

    # 准备输入数据
    questionnaire_json = json.dumps(state.get("questionnaire", {}), ensure_ascii=False, indent=2)
    basic_stats_json = json.dumps(state.get("basic_stats", {}), ensure_ascii=False, indent=2)
    detailed_insights_json = json.dumps(state.get("analysis_insights", {}).get("detailed_insights", []), ensure_ascii=False, indent=2)
    image_guide_json = json.dumps(state.get("image_insertion_guide", []), ensure_ascii=False, indent=2)
    selected_methods = "K-Means 聚类、孤立森林异常检测、相关性分析、Random Forest 特征贡献度评估、LDA 主题建模、语义向量聚类 (Embedding + K-Means)"

    # 全局设定
    system_setup = {
        "role": "system", 
        "content": "你是一位严谨的高级学术研究员和数据分析师。你的任务是逐步撰写一篇关于《企业内部用户中心模块的移动端适配体验调研》的高质量学术报告。请保持语言客观、学术、严谨，不需要任何寒暄或解释，直接输出正文内容。"
    }

    report_sections = []
    messages = [system_setup]

    # --- 阶段 1: 问卷设计 ---
    print(">>> 阶段 1: 撰写问卷设计章节...")
    prompt_1 = f"""
【任务】
请撰写报告的第一部分：## 一、 问卷设计。

【输入数据】
这是我们使用的问卷结构：
{questionnaire_json}

【写作要求】
1. 介绍问卷的整体维度（如人口统计、核心体验量表、开放性痛点收集）。
2. 分析为什么采用这些特定的题型（例如：采用 5 分李克特量表是为了量化满意度，开放题是为了捕获具体场景等）。
3. 字数控制在 300-400 字左右。
"""
    messages.append({"role": "user", "content": prompt_1})
    response_1 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message.content
    report_sections.append(response_1)
    messages.append({"role": "assistant", "content": response_1})

    # --- 阶段 2: 调查结果 ---
    print(">>> 阶段 2: 撰写调查结果章节...")
    prompt_2 = f"""
【任务】
请结合上下文，继续撰写报告的第二部分：## 二、 调查结果。

【输入数据】
这是基于大量有效模拟样本得出的基础统计分布数据：
{basic_stats_json}

【写作要求】
1. 概述样本的总体规模（5000份）。
2. 简明扼要地描述人口统计学特征的分布规律（例如：主要用户群体的年龄段、岗位层级分布、使用的移动设备系统占比等）。
3. 只需要客观陈述数据分布，不需要深入解释原因。
4. 字数控制在 300 字左右。
"""
    messages.append({"role": "user", "content": prompt_2})
    response_2 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message.content
    report_sections.append(response_2)
    messages.append({"role": "assistant", "content": response_2})

    # --- 阶段 3: 分析方式介绍 ---
    print(">>> 阶段 3: 撰写分析方式介绍章节...")
    prompt_3 = f"""
【任务】
请结合上下文，继续撰写报告的第三部分：## 三、 分析方式介绍。

【输入数据】
本次研究主要采用了以下机器学习与统计学分析方法：
{selected_methods}

【写作要求】
1. 逐一介绍上述分析方法的原理。
2. 重点论述将这些方法应用于本次“移动端体验调研”的【学术合理性】和【业务价值】（例如：为什么用 K-Means 能更好地给用户分群？为什么用异常检测找极度不满的用户很重要？）。
3. 语言要体现出数据科学的专业性。
"""
    messages.append({"role": "user", "content": prompt_3})
    response_3 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message.content
    report_sections.append(response_3)
    messages.append({"role": "assistant", "content": response_3})

    # --- 阶段 4: 分析结果 (核心插图环节) ---
    print(">>> 阶段 4: 撰写核心分析结果与图表展示...")
    prompt_4 = f"""
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
"""
    messages.append({"role": "user", "content": prompt_4})
    response_4 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message.content
    report_sections.append(response_4)
    messages.append({"role": "assistant", "content": response_4})

    # --- 阶段 5: 总结与建议 ---
    print(">>> 阶段 5: 撰写总结与建议...")
    prompt_5 = f"""
【任务】
请结合我们刚刚完成的所有章节内容，撰写报告的最后一部分：## 五、 总结与建议。

【写作要求】
1. 凝练前文所有的核心业务洞察（一两句话概括痛点）。
2. 提出 3-4 条极具针对性、可落地的【移动端用户中心适配改进建议】（例如：针对高龄干部的适老化改造、针对弱网环境的离线缓存策略等）。
3. 结尾拔高立意，说明提升用户体验对企业内部效率的整体价值。
4. 字数控制在 400-500 字。
"""
    messages.append({"role": "user", "content": prompt_5})
    response_5 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message.content
    report_sections.append(response_5)
    
    # 拼合最终产物
    full_report = "\n\n".join(report_sections)
    
    # 保存报告
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "thesis_draft.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"✅ 调研报告初稿已生成: {report_path}")

    return {
        "thesis_draft": full_report,
        "current_step": "writer_agent"
    }