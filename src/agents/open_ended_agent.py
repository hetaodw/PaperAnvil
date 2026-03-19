import json
import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from src.workflow.state import SystemState

load_dotenv()

def open_ended_node(state: SystemState) -> dict:
    """
    深度开放题回答生成节点。
    
    针对每个画像，生成指定份数的深度开放题回答。
    数量由画像稀有度决定（稀有画像分配更多深度样本以增加定性分析的多样性）。
    """
    try:
        personas = state.get("personas", [])
        questionnaire = state.get("questionnaire", {})
        open_questions = questionnaire.get("open_ended", [])
        
        if not personas or not open_questions:
            print("[OpenEnded Agent] 警告: 缺少画像或开放题数据。")
            return {"current_step": "open_ended_agent", "open_ended_detailed_responses": []}

        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )
        
        all_responses = []
        
        for persona in personas:
            name = persona.get("name_tag", "未知")
            proportion = persona.get("proportion", 0.1)
            
            # 修正逻辑：遵循调查实质，占比越高的人群生成更多份数
            # 基础规则：常见 (>20%) -> 8-10份, 普通 (5-20%) -> 3-5份, 稀有 (<5%) -> 1-2份
            if proportion > 0.20:
                n_copies = 10
            elif proportion > 0.05:
                n_copies = 5
            else:
                n_copies = 2
                
            print(f"\n>>> 正在为画像 [{name}] 生成 {n_copies} 份深度开放题回答 (占比: {proportion:.2%})...")
            
            for i in range(n_copies):
                print(f"  - 正在生成第 {i+1}/{n_copies} 份...")
                
                system_prompt = f"""你是一位拥有深厚心理学背景的“沉浸式角色扮演专家”。你擅长完全进入给定的人格画像（Persona），并针对调研问卷中的开放性问题，提供极具真实感、细节丰富且带有强烈个人情绪色彩的深度回答。

# Persona Info
- 姓名标签: {name}
- 年龄: {persona.get('age', '未知')}
- 职业: {persona.get('job', '未知')}
- 性格特点: {persona.get('personality', '未知')}
- 居住地: {persona.get('location', '未知')}

# Task
请根据提供的人设信息和问卷题目，以该角色的第一人称口吻进行回答。

# Questionnaire (Open-ended only)
{json.dumps(open_questions, ensure_ascii=False, indent=2)}

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
}}"""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请以该画像的口吻，深度回答开放性问题。"}
                ]
                
                try:
                    completion = client.chat.completions.create(
                        model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
                        messages=messages,
                        extra_body={"enable_thinking": True} # 开放题需要深度思考以满足细节要求
                    )
                    
                    content = completion.choices[0].message.content.strip()
                    clean_json_str = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
                    
                    resp_json = json.loads(clean_json_str)
                    all_responses.append(resp_json)
                    
                except Exception as e:
                    print(f"    ❌ 生成失败: {e}")
                    continue

        # 保存结果
        output_dir = "data/intermediate"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "open_ended_responses.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_responses, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ 深度开放题回答生成完成，共 {len(all_responses)} 条，已保存至 {output_path}")
        
        return {
            "open_ended_detailed_responses": all_responses,
            "current_step": "open_ended_agent"
        }

    except Exception as e:
        error_msg = f"[OpenEnded Agent] 运行失败: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
