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
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        
        all_responses = []
        
        for persona in personas:
            name = persona.get("name_tag", "未知")
            proportion = persona.get("proportion", 0.1)
            
            # 修正逻辑：遵循调查实质，占比越高的人群生成更多份数
            # 基础规则：常见 (>20%) -> 8-10份, 普通 (5-20%) -> 3-5份, 稀有 (<5%) -> 1-2份
            # 测试阶段，只生成 1 份
            n_copies = 1
                
            print(f"\n>>> 正在为画像 [{name}] 生成 {n_copies} 份深度开放题回答 (占比: {proportion:.2%})...")
            
            for i in range(n_copies):
                print(f"  - 正在生成第 {i+1}/{n_copies} 份...")
                
                prompt_template = state.get("prompts", {}).get("open_ended_system_prompt", "")
                if not prompt_template: prompt_template = "你是 {name}，请沉浸回答。"
                system_prompt = prompt_template.format(
                    name=name, age=persona.get('age', '未知'), job=persona.get('job', '未知'),
                    personality=persona.get('personality', '未知'), location=persona.get('location', '未知'),
                    open_questions_json=json.dumps(open_questions, ensure_ascii=False, indent=2)
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请以该画像的口吻，深度回答开放性问题。"}
                ]
                
                try:
                    completion = client.chat.completions.create(
                        model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
                        messages=messages,
                        extra_body={"enable_thinking": False} # 开放题暂时关闭思维链
                    )
                    
                    msg = completion.choices[0].message
                    if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                        print(f"\n[OpenEnded Agent - {name} 思维链]\n{msg.reasoning_content}\n")
                    
                    content = msg.content.strip()
                    print(f"\n[OpenEnded Agent - {name} LLM 输出]\n{content}\n")
                    
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
