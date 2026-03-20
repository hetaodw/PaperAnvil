import json
import os
import re
import requests
from openai import OpenAI
import dashscope
from dashscope import MultiModalConversation
from src.workflow.state import SystemState

def download_and_save_image(image_url: str, save_path: str):
    """从 URL 下载图片并保存到本地"""
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"❌ 下载图片失败: {e}")
        return False

def plotting_agent_node(state: SystemState) -> dict:
    """
    绘图智能体节点：将分析洞察转化为高质量的 AI 生成信息图表。
    """
    print("\n" + "="*60)
    print("📊 [Plotting Agent] 开始根据分析洞察生成数据信息图表...")
    print("="*60)

    # 局部设置 API Key 以确保在 load_dotenv 之后执行
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    analysis_insights = state.get("analysis_insights", {})
    if not analysis_insights or "visualization_plan" not in analysis_insights:
        print("⚠️ 未发现可视化规划 (visualization_plan)，跳过绘图阶段。")
        return {"current_step": "plotting_agent"}

    # 1. 规划阶段：使用文本大模型生成绘画提示词 (Prompt Engineering)
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY")),
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    model_name = os.getenv("MODEL_NAME", "qwen-plus")

    prompt_template = state.get("prompts", {}).get("plotting_chart_design_prompt", "")
    if not prompt_template: prompt_template = "生成图表提示词：{visualization_plan_json}"
    chart_design_prompt = prompt_template.format(
        visualization_plan_json=json.dumps(analysis_insights.get('visualization_plan', []), ensure_ascii=False)
    )

    try:
        response_1 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": chart_design_prompt}],
            temperature=0.7
        )
        msg_1 = response_1.choices[0].message
        if hasattr(msg_1, 'reasoning_content') and msg_1.reasoning_content:
            print(f"\n[Plotting Agent - 图表规划 思维链]\n{msg_1.reasoning_content}\n")
        
        content = msg_1.content.strip()
        print(f"\n[Plotting Agent - 图表规划 LLM 输出]\n{content}\n")
        
        # 增强型 JSON 提取 (支持 Markdown 或 杂质)
        json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        design_plan = json.loads(content)
        image_prompts = design_plan.get("image_prompts", [])
        insertion_guide = design_plan.get("insertion_guide", [])
    except Exception as e:
        error_msg = f"[Plotting Agent] 图表设计规划失败: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error_logs": [error_msg], "current_step": "error"}

    # 2. 生成阶段：调用 Qwen-Image 2.0 实际绘图
    output_dir = "data/output/images"
    os.makedirs(output_dir, exist_ok=True)
    generated_images = []
    final_insertion_guide = []

    for item in image_prompts:
        img_id = item["image_id"]
        prompt_text = item["prompt"]
        print(f"   -> 正在生成图表 {img_id}...")

        try:
            # 调用百炼 MultiModal API
            response = MultiModalConversation.call(
                api_key=dashscope.api_key,
                model="qwen-image-2.0",
                messages=[{"role": "user", "content": [{"text": prompt_text}]}],
                result_format='message'
            )
            
            if response.status_code == 200:
                response_content = response.output.choices[0].message.content
                # 提取图片 URL
                image_url = next((c['image'] for c in response_content if 'image' in c), None)
                
                if image_url:
                    save_path = os.path.join(output_dir, f"{img_id}.png")
                    if download_and_save_image(image_url, save_path):
                        local_rel_path = f"data/output/images/{img_id}.png"
                        generated_images.append(local_rel_path)
                        
                        # 更新插图指南中的路径
                        for guide in insertion_guide:
                            if guide["image_id"] == img_id:
                                guide["local_path"] = local_rel_path
                                final_insertion_guide.append(guide)
                        print(f"      ✅ 图表已保存为: {save_path}")
            else:
                print(f"      ❌ 绘图失败: {response.code} - {response.message}")
        except Exception as e:
            print(f"      ❌ 绘图 API 异常: {e}")

    # 3. 存档插图指南 (供 Writer Agent 使用)
    if final_insertion_guide:
        guide_path = os.path.join("data/processed", "chart_insertion_guide.json")
        os.makedirs("data/processed", exist_ok=True)
        with open(guide_path, 'w', encoding='utf-8') as f:
            json.dump(final_insertion_guide, f, ensure_ascii=False, indent=2)
        print(f"✅ 插图指南已保存至: {guide_path}")

    return {
        "plot_image_paths": generated_images,
        "image_insertion_guide": final_insertion_guide,
        "current_step": "plotting_agent"
    }