import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.workflow.state import SystemState

load_dotenv()

CHECKPOINT_FILE = "data/intermediate/persona_checkpoint.json"


def save_checkpoint(personas: list, batch_num: int, target_count: int):
    checkpoint = {
        "personas": personas,
        "batch_num": batch_num,
        "target_count": target_count
    }
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(f"💾 断点已保存: 已生成 {len(personas)} 个画像 (第 {batch_num} 批)")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🗑️ 断点文件已清除")


def persona_node(state: SystemState) -> dict:
    """
    用户画像生成节点（分批循环生成模式，支持断点续传）。
    
    根据调研主题和问卷，设计该问卷可能调查的人物真实感的人物画像。
    为了防止长文本截断并提高多样性，采用分批（每批最多 3 个）生成的方式。
    支持 JSON 解析失败时保存断点，下次可从断点继续。
    
    Args:
        state: 全局状态字典，包含调研主题、问卷、画像数量等信息
        
    Returns:
        更新后的状态字典，包含生成的所有用户画像
    """
    try:
        topic = state["topic"]
        questionnaire = state["questionnaire"]
        persona_count = state.get("persona_count", 3)
        resume_from_checkpoint = state.get("resume_persona_checkpoint", False)
        
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
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
        start_batch = 0
        
        checkpoint = load_checkpoint()
        if resume_from_checkpoint and checkpoint:
            all_personas = checkpoint.get("personas", [])
            start_batch = checkpoint.get("batch_num", 0)
            saved_target = checkpoint.get("target_count", persona_count)
            print(f"\n🔄 从断点恢复: 已有 {len(all_personas)} 个画像，从第 {start_batch + 1} 批继续")
            if saved_target != persona_count:
                print(f"⚠️ 注意: 断点目标数 ({saved_target}) 与当前目标数 ({persona_count}) 不同")
        
        print(f"\n>>> 计划生成总画像数: {persona_count}，采用分批模式（每批 {batch_size} 个）")
        
        for batch_start in range(start_batch * batch_size, persona_count, batch_size):
            current_batch_size = min(batch_size, persona_count - batch_start)
            batch_num = (batch_start // batch_size) + 1
            print(f"\n--- 正在生成第 {batch_num} 批画像 ({batch_start + 1}-{min(batch_start + current_batch_size, persona_count)}) ---")
            
            existing_context = ""
            if all_personas:
                existing_names = [p.get("name_tag", "未知") for p in all_personas]
                existing_context = f"\n\n【已存在的画像姓名/标签】: {', '.join(existing_names)}\n请确保新生成的画像在职业、性格、痛点和人口统计特征上与上述已有画像有显著差异，避免重复。"
            
            prompt_template = state.get("prompts", {}).get("persona_system_prompt", "")
            if not prompt_template: prompt_template = "请为人设 {topic} 生成画像。"
            system_prompt = prompt_template.format(
                topic=topic,
                demographics_info=demographics_info,
                likert_info=likert_info,
                open_ended_info=open_ended_info,
                existing_context=existing_context
            )
            
            temperature = 0.8 + (batch_num * 0.05) 
            if temperature > 1.2: temperature = 1.2
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请生成 {current_batch_size} 个全新的、具有差异化的中国本地居民用户画像。"}
            ]
            
            completion = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
                messages=messages,
                temperature=temperature,
                extra_body={"enable_thinking": False},
                stream=True
            )
            
            is_answering = False
            answer_content = ""
            reasoning_content = ""
            
            for chunk in completion:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        is_answering = True
                    answer_content += delta.content
            
            if reasoning_content:
                print(f"\n[Persona Agent - 批次 {batch_num} 思维链]\n{reasoning_content}\n")
            print(f"\n[Persona Agent - 批次 {batch_num} LLM 输出]\n{answer_content}\n")
            
            try:
                batch_json = json.loads(answer_content.strip())
                if "personas" in batch_json:
                    all_personas.extend(batch_json["personas"])
                    print(f"✅ 第 {batch_num} 批生成成功，当前累计: {len(all_personas)}")
                    save_checkpoint(all_personas, batch_num, persona_count)
            except json.JSONDecodeError as e:
                print(f"⚠️ 第 {batch_num} 批 JSON 解析失败: {e}")
                print(f"💾 已保存当前进度 ({len(all_personas)} 个画像) 到断点文件")
                save_checkpoint(all_personas, batch_num, persona_count)
                raise

        if all_personas:
            total_proportion = sum(p.get("proportion", 0) for p in all_personas)
            if total_proportion == 0:
                for p in all_personas:
                    p["proportion"] = 1.0 / len(all_personas)
            else:
                for p in all_personas:
                    p["proportion"] = p.get("proportion", 0) / total_proportion
            
            print(f"\n✅ 所有 {len(all_personas)} 个画像已完成归一化。")

        final_output = {"personas": all_personas}
        intermediate_dir = "data/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        personas_path = os.path.join(intermediate_dir, "personas.json")
        
        with open(personas_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        clear_checkpoint()
        
        return {
            "personas": all_personas,
            "current_step": "persona_agent"
        }
        
    except json.JSONDecodeError as e:
        error_msg = f"[Persona Agent] JSON 解析失败: {str(e)}"
        print(error_msg)
        print("💡 提示: 可设置 resume_persona_checkpoint=True 从断点继续")
        return {"error_logs": [error_msg], "current_step": "error"}
    except Exception as e:
        error_msg = f"[Persona Agent] 运行失败: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
