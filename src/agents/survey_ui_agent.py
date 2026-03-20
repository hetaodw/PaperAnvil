import json
import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from src.workflow.state import SystemState

load_dotenv()

def _clean_markdown(text: str) -> str:
    """清理 Markdown 标签等干扰字符"""
    return re.sub(r'^```html\s*|\s*```$', '', text, flags=re.MULTILINE).strip()

def survey_ui_node(state: SystemState) -> dict:
    """
    调研问卷 UI 生成节点。
    
    输入：问卷结构 (JSON)
    输出：美观、现代、高质感的问卷 HTML 页面
    
    Args:
        state: 全局状态字典，包含问卷结构等信息
    """
    try:
        questionnaire = state.get("questionnaire", {})
        if not questionnaire:
            raise ValueError("State 中没有问卷结构 (questionnaire)")
            
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("MODEL_NAME", "qwen-max") # 使用 qwen-max 作为默认，可能比 qwen3.5-plus 更稳
        api_key = os.getenv("DASHSCOPE_API_KEY")
        
        print(f"--- Debug: API_KEY={api_key[:8]}... BASE_URL={base_url} MODEL={model} ---")
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        prompt_template = state.get("prompts", {}).get("survey_ui_prompt", "")
        if not prompt_template: prompt_template = "请基于该问卷 {questionnaire_json} 生成网页。"
        prompt = prompt_template.format(questionnaire_json=json.dumps(questionnaire, ensure_ascii=False, indent=2))

        print("\n>>> 正在生成问卷 UI HTML...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位顶级前端设计师，只输出单文件 HTML 代码。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            extra_body={"enable_thinking": False}
        )
        
        content = response.choices[0].message.content
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            print(f"\n[Survey UI Agent 思维链]\n{response.choices[0].message.reasoning_content}\n")
            
        html_code = _clean_markdown(content)
        
        # 保存文件
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "survey.html")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_code)
            
        print(f"✅ 问卷 UI 已生成并保存至: {file_path}")
        
        return {
            "current_step": "survey_ui_agent"
        }
        
    except Exception as e:
        error_msg = f"[Survey UI Agent] 运行失败: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
