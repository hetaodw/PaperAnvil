import json
import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from src.workflow.state import SystemState

load_dotenv()

def _clean_json_string(json_str: str) -> str:
    """清理 Markdown 标签等干扰字符"""
    return re.sub(r'^```json\s*|\s*```$', '', json_str, flags=re.MULTILINE).strip()

def survey_node(state: SystemState) -> dict:
    """
    调研问卷设计节点（双阶段增强版）。
    
    第一阶段：生成问卷标题、人口统计和前 10 道李克特量表题。
    第二阶段：生成后 10 道李克特量表题和开放性问题。
    
    Args:
        state: 全局状态字典，包含调研主题等信息
    """
    try:
        topic = state["topic"]
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        
        # --- 第一阶段：基础与前 10 题 ---
        print("\n>>> 正在进行第一阶段问卷生成 (标题、人口统计、L1-L10)...")
        prompt_template_1 = state.get("prompts", {}).get("survey_phase1_prompt", "")
        if not prompt_template_1:
            prompt_template_1 = f"你一位资深研究员。请为 {topic} 设计问卷第一部分。" # Fallback (shouldn't be reached)
        prompt_1 = prompt_template_1.format(topic=topic)
        
        response_1 = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
            messages=[{"role": "user", "content": prompt_1}],
            extra_body={"enable_thinking": True}
        )
        
        msg_1 = response_1.choices[0].message
        if hasattr(msg_1, 'reasoning_content') and msg_1.reasoning_content:
            print(f"\n[Survey Agent - Phase 1 思维链]\n{msg_1.reasoning_content}\n")
        print(f"\n[Survey Agent - Phase 1 LLM 输出]\n{msg_1.content}\n")
        
        json_1 = json.loads(_clean_json_string(response_1.choices[0].message.content))
        
        # --- 第二阶段：后 10 题与开放题 ---
        print(">>> 正在进行第二阶段问卷生成 (L11-L20、开放题)...")
        context_1 = json.dumps(json_1.get("likert_scales", []), ensure_ascii=False)
        prompt_template_2 = state.get("prompts", {}).get("survey_phase2_prompt", "")
        if not prompt_template_2: prompt_template_2 = f"请针对 {topic} 生成第二部分问卷。"
        prompt_2 = prompt_template_2.format(context_1=context_1, topic=topic)

        response_2 = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
            messages=[{"role": "user", "content": prompt_2}],
            extra_body={"enable_thinking": False}
        )
        
        msg_2 = response_2.choices[0].message
        if hasattr(msg_2, 'reasoning_content') and msg_2.reasoning_content:
            print(f"\n[Survey Agent - Phase 2 思维链]\n{msg_2.reasoning_content}\n")
        print(f"\n[Survey Agent - Phase 2 LLM 输出]\n{msg_2.content}\n")
        
        json_2 = json.loads(_clean_json_string(response_2.choices[0].message.content))
        
        # --- 合并结果 ---
        final_questionnaire = {
            "survey_title": json_1.get("survey_title", "未命名问卷"),
            "demographics": json_1.get("demographics", []),
            "likert_scales": json_1.get("likert_scales", []) + json_2.get("likert_scales", []),
            "open_ended": json_2.get("open_ended", [])
        }
        
        print(f"✅ 问卷生成完成，共有 {len(final_questionnaire['likert_scales'])} 道李克特题。")
        
        # 保存
        intermediate_dir = "data/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        with open(os.path.join(intermediate_dir, "questionnaire.json"), 'w', encoding='utf-8') as f:
            json.dump(final_questionnaire, f, ensure_ascii=False, indent=2)
            
        return {
            "questionnaire": final_questionnaire,
            "current_step": "survey_agent"
        }
        
    except Exception as e:
        error_msg = f"[Survey Agent] 运行失败: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
