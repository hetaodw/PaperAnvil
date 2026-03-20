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

    prompt_template_setup = state.get("prompts", {}).get("writer_system_setup", "")
    if not prompt_template_setup: prompt_template_setup = "你是高级研究员。请撰写《{topic}》。"
    system_setup = {
        "role": "system", 
        "content": prompt_template_setup.format(topic=state.get('topic', '调研报告'))
    }

    report_sections = []
    messages = [system_setup]

    # --- 阶段 1: 问卷设计 ---
    print(">>> 阶段 1: 撰写问卷设计章节...")
    prompt_template_1 = state.get("prompts", {}).get("writer_prompt_1", "")
    if not prompt_template_1: prompt_template_1 = "撰写问卷设计：\n{questionnaire_json}"
    prompt_1 = prompt_template_1.format(questionnaire_json=questionnaire_json)
    messages.append({"role": "user", "content": prompt_1})
    msg_1 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message
    if hasattr(msg_1, 'reasoning_content') and msg_1.reasoning_content:
        print(f"\n[Writer Agent - 阶段 1 思维链]\n{msg_1.reasoning_content}\n")
    response_1 = msg_1.content
    print(f"\n[Writer Agent - 阶段 1 LLM 输出]\n{response_1}\n")
    report_sections.append(response_1)
    messages.append({"role": "assistant", "content": response_1})

    # --- 阶段 2: 调查结果 ---
    print(">>> 阶段 2: 撰写调查结果章节...")
    prompt_template_2 = state.get("prompts", {}).get("writer_prompt_2", "")
    if not prompt_template_2: prompt_template_2 = "撰写调查结果：\n{basic_stats_json}"
    prompt_2 = prompt_template_2.format(basic_stats_json=basic_stats_json)
    messages.append({"role": "user", "content": prompt_2})
    msg_2 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message
    if hasattr(msg_2, 'reasoning_content') and msg_2.reasoning_content:
        print(f"\n[Writer Agent - 阶段 2 思维链]\n{msg_2.reasoning_content}\n")
    response_2 = msg_2.content
    print(f"\n[Writer Agent - 阶段 2 LLM 输出]\n{response_2}\n")
    report_sections.append(response_2)
    messages.append({"role": "assistant", "content": response_2})

    # --- 阶段 3: 分析方式介绍 ---
    print(">>> 阶段 3: 撰写分析方式介绍章节...")
    prompt_template_3 = state.get("prompts", {}).get("writer_prompt_3", "")
    if not prompt_template_3: prompt_template_3 = "撰写分析方式：\n{selected_methods}"
    prompt_3 = prompt_template_3.format(selected_methods=selected_methods)
    messages.append({"role": "user", "content": prompt_3})
    msg_3 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message
    if hasattr(msg_3, 'reasoning_content') and msg_3.reasoning_content:
        print(f"\n[Writer Agent - 阶段 3 思维链]\n{msg_3.reasoning_content}\n")
    response_3 = msg_3.content
    print(f"\n[Writer Agent - 阶段 3 LLM 输出]\n{response_3}\n")
    report_sections.append(response_3)
    messages.append({"role": "assistant", "content": response_3})

    # --- 阶段 4: 分析结果 (核心插图环节) ---
    print(">>> 阶段 4: 撰写核心分析结果与图表展示...")
    prompt_template_4 = state.get("prompts", {}).get("writer_prompt_4", "")
    if not prompt_template_4: prompt_template_4 = "撰写分析结果：\n{detailed_insights_json}\n图片指南：{image_guide_json}"
    prompt_4 = prompt_template_4.format(
        detailed_insights_json=detailed_insights_json,
        image_guide_json=image_guide_json
    )
    messages.append({"role": "user", "content": prompt_4})
    msg_4 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message
    if hasattr(msg_4, 'reasoning_content') and msg_4.reasoning_content:
        print(f"\n[Writer Agent - 阶段 4 思维链]\n{msg_4.reasoning_content}\n")
    response_4 = msg_4.content
    print(f"\n[Writer Agent - 阶段 4 LLM 输出]\n{response_4}\n")
    report_sections.append(response_4)
    messages.append({"role": "assistant", "content": response_4})

    # --- 阶段 5: 中国现状与典型案例分析 ---
    print(">>> 阶段 5: 撰写中国现状与典型案例分析...")
    prompt_template_5 = state.get("prompts", {}).get("writer_prompt_5", "")
    if not prompt_template_5: prompt_template_5 = "撰写中国现状：\n{topic}"
    prompt_5 = prompt_template_5.format(topic=state.get('topic', ''))
    messages.append({"role": "user", "content": prompt_5})
    msg_5 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message
    if hasattr(msg_5, 'reasoning_content') and msg_5.reasoning_content:
        print(f"\n[Writer Agent - 阶段 5 思维链]\n{msg_5.reasoning_content}\n")
    response_5 = msg_5.content
    print(f"\n[Writer Agent - 阶段 5 LLM 输出]\n{response_5}\n")
    report_sections.append(response_5)
    messages.append({"role": "assistant", "content": response_5})

    # --- 阶段 6: 总结与可借鉴经验建议 ---
    print(">>> 阶段 6: 撰写总结与可借鉴经验建议...")
    prompt_template_6 = state.get("prompts", {}).get("writer_prompt_6", "")
    if not prompt_template_6: prompt_template_6 = "写总结：\n{topic}"
    prompt_6 = prompt_template_6.format(topic=state.get('topic', ''))
    messages.append({"role": "user", "content": prompt_6})
    msg_6 = client.chat.completions.create(model=model_name, messages=messages).choices[0].message
    if hasattr(msg_6, 'reasoning_content') and msg_6.reasoning_content:
        print(f"\n[Writer Agent - 阶段 6 思维链]\n{msg_6.reasoning_content}\n")
    response_6 = msg_6.content
    print(f"\n[Writer Agent - 阶段 6 LLM 输出]\n{response_6}\n")
    report_sections.append(response_6)
    
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