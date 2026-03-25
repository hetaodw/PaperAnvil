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

def text_to_survey_node(state: SystemState) -> dict:
    """
    文本转问卷 Agent。
    
    输入：用户粘贴的非结构化/半结构化问卷文本
    输出：标准的 questionnaire JSON 结构
    
    Args:
        state: 全局状态字典，包含调研主题和原始文本等信息
    """
    try:
        topic = state.get("topic", "未命名主题")
        input_text = state.get("input_text", "")
        
        if not input_text:
            raise ValueError("State 中没有获取到 input_text")
            
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        
        print("\n>>> 正在将原始文本转换为标准问卷 JSON...")
        prompt_template = state.get("prompts", {}).get("text_to_survey_prompt", "")
        if not prompt_template:
            # Fallback
            prompt_template = "请将以下文本转换为相应的 JSON 格式。主题是 {topic}。\n\n文本内容：\n{input_text}"
            
        prompt = prompt_template.format(topic=topic, input_text=input_text)
        
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
            messages=[{"role": "user", "content": prompt}],
            extra_body={"enable_thinking": True}
        )
        
        msg = response.choices[0].message
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            print(f"\n[Text-to-Survey Agent 思维链]\n{msg.reasoning_content}\n")
        print(f"\n[Text-to-Survey Agent LLM 输出]\n{msg.content}\n")
        
        final_questionnaire = json.loads(_clean_json_string(msg.content))
        
        # 补齐默认字段，以防 LLM 漏掉
        if "survey_title" not in final_questionnaire:
            final_questionnaire["survey_title"] = topic
        if "demographics" not in final_questionnaire:
            final_questionnaire["demographics"] = []
        if "likert_scales" not in final_questionnaire:
            final_questionnaire["likert_scales"] = []
        if "open_ended" not in final_questionnaire:
            final_questionnaire["open_ended"] = []
            
        print(f"✅ 问卷转换完成，提取到 {len(final_questionnaire['likert_scales'])} 道李克特题。")
        
        # 保存
        intermediate_dir = "data/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        with open(os.path.join(intermediate_dir, "questionnaire.json"), 'w', encoding='utf-8') as f:
            json.dump(final_questionnaire, f, ensure_ascii=False, indent=2)
            
        return {
            "questionnaire": final_questionnaire,
            "current_step": "text_to_survey_agent"
        }
        
    except Exception as e:
        error_msg = f"[Text-to-Survey Agent] 运行失败: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
