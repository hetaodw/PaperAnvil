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
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )
        
        # --- 第一阶段：基础与前 10 题 ---
        print("\n>>> 正在进行第一阶段问卷生成 (标题、人口统计、L1-L10)...")
        prompt_1 = f"""你是一位资深的社会学研究员。请针对主题“{topic}”设计问卷的第一部分。
要求包含：
1. 问卷标题。
2. 3-5 道人口统计题 (id: d1, d2...)。
3. 前 10 道李克特量表题 (id: l1 到 l10，1-5分，1为极差/极不满意，5为极好/极其满意)。

只输出 JSON，格式如下：
{{
  "survey_title": "...",
  "demographics": [{{ "id": "d1", "question": "...", "options": [...] }}],
  "likert_scales": [{{ "id": "l1", "question": "...", "scale_range": [1, 5], "labels": {{ "1": "极差", "5": "极好" }} }}]
}}"""
        
        response_1 = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
            messages=[{"role": "user", "content": prompt_1}],
            extra_body={"enable_thinking": True}
        )
        
        json_1 = json.loads(_clean_json_string(response_1.choices[0].message.content))
        
        # --- 第二阶段：后 10 题与开放题 ---
        print(">>> 正在进行第二阶段问卷生成 (L11-L20、开放题)...")
        context_1 = json.dumps(json_1.get("likert_scales", []), ensure_ascii=False)
        prompt_2 = f"""你已经生成了以下 10 道题目：{context_1}。
现在请继续针对主题“{topic}”生成问卷的第二部分。
要求包含：
1. 另外 10 道互不重复的李克特量表题 (id: l11 到 l20，保持 l11, l12... 的顺序)。
2. 2 道开放性问题 (id: o1, o2)。

只输出 JSON，格式如下：
{{
  "likert_scales": [{{ "id": "l11", "question": "...", "scale_range": [1, 5], "labels": {{ "1": "极差", "5": "极好" }} }}],
  "open_ended": [{{ "id": "o1", "question": "..." }}]
}}"""

        response_2 = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
            messages=[{"role": "user", "content": prompt_2}],
            extra_body={"enable_thinking": True}
        )
        
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
