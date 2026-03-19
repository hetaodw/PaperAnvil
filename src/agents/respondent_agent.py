import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from src.workflow.state import SystemState

load_dotenv()


def respondent_node(state: SystemState) -> dict:
    """
    角色扮演答题节点。
    
    遍历 state["personas"] 列表，针对每一个画像调用大模型进行沉浸式答题。
    
    Args:
        state: 全局状态字典，包含画像列表和问卷结构
        
    Returns:
        更新后的状态字典，包含所有种子回答 seed_responses
    """
    try:
        personas = state.get("personas", [])
        questionnaire = state.get("questionnaire", {})
        
        if not personas:
            print("[Respondent Agent] 警告: 状态中没有画像数据。")
            return {"current_step": "respondent_agent", "seed_responses": []}
            
        if not questionnaire:
            print("[Respondent Agent] 警告: 状态中没有问卷数据。")
            return {"current_step": "respondent_agent", "seed_responses": []}

        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )
        
        all_seeds = []
        
        # 问卷结构文本化，方便注入 Prompt
        survey_context = json.dumps(questionnaire, ensure_ascii=False, indent=2)
        
        for persona in personas:
            name = persona.get("name_tag", "未知")
            age = persona.get("age", "未知")
            job = persona.get("job", "未知")
            personality = persona.get("personality", "普通")
            location = persona.get("location", "未知")
            
            print(f"\n{'='*60}")
            print(f"📋 获取到画像信息：")
            print(f"  姓名：{name}")
            print(f"  年龄：{age}")
            print(f"  职业：{job}")
            print(f"  性格：{personality}")
            print(f"  居住地：{location}")
            print(f"{'='*60}")
            
            print(f"\n>>> 正在为画像 [{name}] 生成回答...")
            
            system_prompt = f"""你现在是：{name}。
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

注意：只输出 JSON 对象，不要包含任何 Markdown 格式的包裹符号（如 ```json）。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请根据你的设定，完成问卷回答。"}
            ]
            
            completion = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
                messages=messages,
                extra_body={"enable_thinking": False}, # 答题过程追求快速准确
                stream=False
            )
            
            answer_content = completion.choices[0].message.content.strip()
            
            # 清洗 Markdown 标签
            clean_json_str = re.sub(r'^```json\s*|\s*```$', '', answer_content, flags=re.MULTILINE).strip()
            
            try:
                raw_response = json.loads(clean_json_str)
                
                # 数据清洗与字段精简
                pruned_seed = {
                    "persona_name": name,
                    "responses": raw_response.get("responses", raw_response) # 兼容 AI 可能直接输出 key-value 的情况
                }
                all_seeds.append(pruned_seed)
                print(f"✅ 画像 [{name}] 回答已保存。")
                
            except json.JSONDecodeError as e:
                print(f"❌ 画像 [{name}] JSON 解析失败: {str(e)}")
                # 记录到错误日志但不中断全局
                continue

        # 存档
        intermediate_dir = "data/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        seed_path = os.path.join(intermediate_dir, "seed_responses.json")
        
        with open(seed_path, 'w', encoding='utf-8') as f:
            json.dump(all_seeds, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ 所有画像答题完成，共 {len(all_seeds)} 条记录，已保存至 {seed_path}")

        return {
            "seed_responses": all_seeds,
            "current_step": "respondent_agent"
        }
        
    except Exception as e:
        error_msg = f"[Respondent Agent] 运行发生致命错误: {str(e)}"
        print(error_msg)
        return {
            "error_logs": [error_msg],
            "current_step": "error"
        }
