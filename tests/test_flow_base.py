import os
import sys
import json
from dotenv import load_dotenv

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.state import SystemState
from src.agents.survey_agent import survey_node
from src.agents.persona_agent import persona_node
from src.agents.respondent_agent import respondent_node

def test_integration_flow():
    # 1. 环境准备：加载 .env 环境变量
    load_dotenv()
    
    # 2. 初始化状态
    initial_state: SystemState = {
        "topic": "企业内部用户中心模块的移动端适配体验调研",
        "persona_count": 3,  # 本次测试减少为3个，节约时间
        "thread_id": "test_integration_001",
        "questionnaire": {},
        "personas": [],
        "seed_responses": [],
        "plot_image_paths": [],
        "analysis_insights": {},
        "thesis_draft": "",
        "error_logs": [],
        "current_step": "start"
    }
    
    print(f"\n>>> 开始集成测试: {initial_state['topic']}")
    print(f">>> Thread ID: {initial_state['thread_id']}\n")

    # 3. 执行流水线 - 第一步：Survey Agent
    print("--- 步骤 1: 运行 Survey Agent ---")
    update_1 = survey_node(initial_state)
    updated_state_1 = {**initial_state, **update_1}
    
    if "error_logs" in update_1 and update_1["error_logs"]:
        print(f"❌ Survey Agent 报错: {update_1['error_logs']}")
        return
    print("✅ Survey Agent 运行成功")
    
    # 4. 执行流水线 - 第二步：Persona Agent
    print("\n--- 步骤 2: 运行 Persona Agent ---")
    update_2 = persona_node(updated_state_1)
    updated_state_2 = {**updated_state_1, **update_2}
    
    if "personas" not in updated_state_2 or not updated_state_2["personas"]:
        print("❌ Persona Agent 未生成画像数据")
        return
    print("✅ Persona Agent 运行成功")

    # 5. 执行流水线 - 第三步：Respondent Agent
    print("\n--- 步骤 3: 运行 Respondent Agent ---")
    update_3 = respondent_node(updated_state_2)
    final_state = {**updated_state_2, **update_3}
    
    if "seed_responses" not in final_state or not final_state["seed_responses"]:
        print("❌ Respondent Agent 未生成种子回答数据")
        return
    print("✅ Respondent Agent 运行成功")

    # 6. 验证与输出
    print("\n" + "="*30 + " 验证与报告 " + "="*30)
    
    q_file = "data/intermediate/questionnaire.json"
    p_file = "data/intermediate/personas.json"
    s_file = "data/intermediate/seed_responses.json"
    
    for f in [q_file, p_file, s_file]:
        if os.path.exists(f):
            print(f"文件已生成: {f}")
        else:
            print(f"⚠️ 警告: 文件缺失 {f}")

    # 打印报告
    questionnaire = final_state.get("questionnaire", {})
    survey_title = questionnaire.get("survey_title", "未命名问卷")
    print(f"\n【问卷标题】: {survey_title}")
    
    personas = final_state.get("personas", [])
    print(f"\n【画像列表】(共 {len(personas)} 个):")
    total_proportion = 0
    
    for i, p in enumerate(personas, 1):
        name = p.get("name_tag", "未知")
        proportion = p.get("proportion", 0)
        total_proportion += proportion
        print(f"  {i}. {name} (占比: {proportion:.2%})")

    # 打印种子回答摘要
    seeds = final_state.get("seed_responses", [])
    print(f"\n【种子回答摘要】(共 {len(seeds)} 条):")
    for i, seed in enumerate(seeds, 1):
        name = seed.get("persona_name", "未知")
        res_count = len(seed.get("responses", {}))
        print(f"  {i}. 来自 {name} 的回答 (包含 {res_count} 个题目的回答)")

    # 统计验证
    print(f"\n【统计验证】:")
    print(f"  所有画像占比总和: {total_proportion:.4f}")
    if abs(total_proportion - 1.0) < 0.001:
        print("  ✅ 比例验证通过 (Sum = 1.0)")
    else:
        print(f"  ❌ 比例验证失败 (Sum = {total_proportion:.4f})")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_integration_flow()
