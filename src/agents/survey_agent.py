import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.workflow.state import SystemState

load_dotenv()


def survey_node(state: SystemState) -> dict:
    """
    调研问卷设计节点。
    
    根据给定的主题，设计一份专业的量化与定性结合的调研问卷。
    问卷包含人口统计、李克特量表和开放性问题。
    
    Args:
        state: 全局状态字典，包含调研主题等信息
        
    Returns:
        更新后的状态字典，包含生成的问卷结构
    """
    try:
        topic = state["topic"]
        
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )
        
        system_prompt = """你是一位资深的社会学研究员和系统用户体验专家。你的任务是根据给定的主题，设计一份专业的调研问卷。
问卷必须包含三种题型：
人口统计与背景（如：年龄段、岗位层级、系统使用频率）。

核心体验量表（必须是 1-5 分的李克特量表，例如：5代表极其满意/毫无卡顿，1代表极其不满/严重卡顿，至少 5 题）。

开放性问题（至少 2 题，用于收集痛点和具体场景描述）。

【输出强制要求】
你必须且只能输出一个合法的 JSON 对象，不要包含任何 Markdown 标记（如 ```json），不要有任何前言或后语。
JSON 结构必须如下：
{
"survey_title": "问卷标题",
"demographics": [{"id": "d1", "question": "...", "options": ["...", "..."]}],
"likert_scales": [{"id": "l1", "question": "...", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}}],
"open_ended": [{"id": "o1", "question": "..."}]
}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"调研主题：{topic}"}
        ]
        
        completion = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
            messages=messages,
            extra_body={"enable_thinking": True},
            stream=True
        )
        
        is_answering = False
        print("\n" + "=" * 20 + "思考过程" + "=" * 20)
        
        reasoning_content = ""
        answer_content = ""
        
        for chunk in completion:
            delta = chunk.choices[0].delta
            
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    reasoning_content += delta.reasoning_content
                    print(delta.reasoning_content, end="", flush=True)
            
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                answer_content += delta.content
                print(delta.content, end="", flush=True)
        
        json_str = answer_content.strip()
        
        parsed_json = json.loads(json_str)
        
        intermediate_dir = "data/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        questionnaire_path = os.path.join(intermediate_dir, "questionnaire.json")
        
        with open(questionnaire_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=2)
        
        return {
            "questionnaire": parsed_json,
            "current_step": "survey_agent"
        }
        
    except json.JSONDecodeError as e:
        error_msg = f"[Survey Agent] JSON 解析失败: {str(e)}"
        print(error_msg)
        return {
            "error_logs": [error_msg],
            "current_step": "error"
        }
    except Exception as e:
        error_msg = f"[Survey Agent] 运行失败: {str(e)}"
        print(error_msg)
        return {
            "error_logs": [error_msg],
            "current_step": "error"
        }
