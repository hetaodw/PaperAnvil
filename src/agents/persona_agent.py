import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.workflow.state import SystemState

load_dotenv()


def persona_node(state: SystemState) -> dict:
    """
    用户画像生成节点（分批循环生成模式）。
    
    根据调研主题和问卷，设计该问卷可能调查的人物真实感的人物画像。
    为了防止长文本截断并提高多样性，采用分批（每批最多 3 个）生成的方式。
    
    Args:
        state: 全局状态字典，包含调研主题、问卷、画像数量等信息
        
    Returns:
        更新后的状态字典，包含生成的所有用户画像
    """
    try:
        topic = state["topic"]
        questionnaire = state["questionnaire"]
        persona_count = state.get("persona_count", 3)
        
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )
        
        demographics_info = ""
        if "demographics" in questionnaire:
            for demo in questionnaire["demographics"]:
                demographics_info += f"\n- {demo.get('id', '')}: {demo.get('question', '')}\n  选项: {', '.join(demo.get('options', []))}"
        
        likert_info = ""
        if "likert_scales" in questionnaire:
            for likert in questionnaire["likert_scales"]:
                likert_info += f"\n- {likert.get('id', '')}: {likert.get('question', '')}"
        
        open_ended_info = ""
        if "open_ended" in questionnaire:
            for open_q in questionnaire["open_ended"]:
                open_ended_info += f"\n- {open_q.get('id', '')}: {open_q.get('question', '')}"
        
        all_personas = []
        batch_size = 3
        
        print(f"\n>>> 计划生成总画像数: {persona_count}，采用分批模式（每批 {batch_size} 个）")
        
        for batch_start in range(0, persona_count, batch_size):
            current_batch_size = min(batch_size, persona_count - batch_start)
            batch_num = (batch_start // batch_size) + 1
            print(f"\n--- 正在生成第 {batch_num} 批画像 ({batch_start + 1}-{min(batch_start + current_batch_size, persona_count)}) ---")
            
            existing_context = ""
            if all_personas:
                existing_names = [p.get("name_tag", "未知") for p in all_personas]
                existing_context = f"\n\n【已存在的画像姓名/标签】: {', '.join(existing_names)}\n请确保新生成的画像在职业、性格、痛点和人口统计特征上与上述已有画像有显著差异，避免重复。"
            
            system_prompt = f"""你是一位融合了心理学、社会学与大数据建模能力的资深画像架构师。
你的任务是根据调研主题和问卷，设计出极其鲜活、具有'数字孪生'真实感的人物画像。

【调研主题】
{topic}

【问卷结构】
{demographics_info}
{likert_info}
{open_ended_info}{existing_context}

【每个画像必须包含以下字段】
- name_tag: 姓名风格+标签（如：后端开发-王强）
- gender: 性别（男/女）
- age: 具体年龄（整数）
- job: 具体职业（如：中关村大厂程序员、退休教师等）
- personality: 详细性格描述（如：对系统卡顿极度焦虑、追求极致效率、技术保守派等）
- location: 具体的居住城市及区域（如：北京海淀、上海徐汇）
- proportion: 该类人群在 5000 条样本中的原始预计权重（后续会自动归一化）
- demographics_fixed: 必须匹配问卷中 demographics 的 id，并从对应的 options 中选择一个符合该人设的文本
- likert_distribution: 针对问卷中每个李克特量表题 id，设定其得分均值 mu (1.0-5.0) 和标准差 sigma (0.3-0.9)
- open_ended_samples: 针对问卷中的开放题 id，以该人物的口吻提供 2 条极其写实、带有性格色彩的口语化回答

【输出强制要求】
只输出合法的 JSON 对象，不包含 Markdown 代码块。
JSON 结构：{{"personas": [{{画像1}}, {{画像2}}, ... ]}}"""
            
            # 使用稍高的温度以增加随机性和多样性
            temperature = 0.8 + (batch_num * 0.05) 
            if temperature > 1.2: temperature = 1.2
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请生成 {current_batch_size} 个全新的、具有差异化的用户画像。"}
            ]
            
            completion = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
                messages=messages,
                temperature=temperature,
                extra_body={"enable_thinking": True},
                stream=True
            )
            
            is_answering = False
            answer_content = ""
            
            print(f">>> 第 {batch_num} 批思考中...")
            for chunk in completion:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        is_answering = True
                    answer_content += delta.content
            
            batch_json = json.loads(answer_content.strip())
            if "personas" in batch_json:
                all_personas.extend(batch_json["personas"])
                print(f"✅ 第 {batch_num} 批生成成功，当前累计: {len(all_personas)}")

        # 全局归一化处理
        if all_personas:
            total_proportion = sum(p.get("proportion", 0) for p in all_personas)
            if total_proportion == 0:
                for p in all_personas:
                    p["proportion"] = 1.0 / len(all_personas)
            else:
                for p in all_personas:
                    p["proportion"] = p.get("proportion", 0) / total_proportion
            
            print(f"\n✅ 所有 {len(all_personas)} 个画像已完成归一化。")

        # 保存结果
        final_output = {"personas": all_personas}
        intermediate_dir = "data/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        personas_path = os.path.join(intermediate_dir, "personas.json")
        
        with open(personas_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        return {
            "personas": all_personas,
            "current_step": "persona_agent"
        }
        
    except json.JSONDecodeError as e:
        error_msg = f"[Persona Agent] JSON 解析失败: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
    except Exception as e:
        error_msg = f"[Persona Agent] 运行失败: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
