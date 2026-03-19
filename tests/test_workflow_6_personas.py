import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.state import SystemState
from src.agents.survey_agent import survey_node
from src.agents.persona_agent import persona_node
from src.agents.respondent_agent import respondent_node
from src.agents.data_expansion_agent import data_expansion_node
from src.tools.csv_to_xlsx import convert_csv_to_xlsx

def test_workflow_6_personas():
    # 1. 环境准备：加载 .env 环境变量
    load_dotenv()
    
    # 2. 初始化状态
    initial_state: SystemState = {
        "topic": "针对 AIGC 辅助编程工具（如 GitHub Copilot, Cursor）在企业研发团队中的使用体验与生产力影响调研",
        "persona_count": 6,  # 按照要求生成 6 个人格
        "thread_id": "test_workflow_6_personas_001",
        "questionnaire": {},
        "personas": [],
        "seed_responses": [],
        "plot_image_paths": [],
        "analysis_insights": {},
        "thesis_draft": "",
        "error_logs": [],
        "current_step": "start"
    }
    
    print(f"\n>>> 开始 6 个人格工作流测试: {initial_state['topic']}")
    print(f">>> Thread ID: {initial_state['thread_id']}\n")

    # 3. 执行流水线 - 第一步：Survey Agent (问卷生成)
    print("--- 步骤 1: 运行 Survey Agent (问卷生成) ---")
    update_1 = survey_node(initial_state)
    if "error_logs" in update_1 and update_1["error_logs"]:
        print(f"❌ Survey Agent 报错: {update_1['error_logs']}")
        return
    updated_state_1 = {**initial_state, **update_1}
    print("✅ Survey Agent 运行成功")
    
    # 4. 执行流水线 - 第二步：Persona Agent (人格生成)
    print("\n--- 步骤 2: 运行 Persona Agent (人格生成) ---")
    update_2 = persona_node(updated_state_1)
    if "error_logs" in update_2 and update_2["error_logs"]:
        print(f"❌ Persona Agent 报错: {update_2['error_logs']}")
        return
    updated_state_2 = {**updated_state_1, **update_2}
    
    personas = updated_state_2.get("personas", [])
    if len(personas) != 6:
        print(f"⚠️ 警告: 预期生成 6 个画像，实际生成 {len(personas)} 个")
    else:
        print(f"✅ Persona Agent 运行成功，生成了 {len(personas)} 个画像")

    # 5. 执行流水线 - 第三步：Respondent Agent (标准答案回答)
    print("\n--- 步骤 3: 运行 Respondent Agent (标准答案回答) ---")
    update_3 = respondent_node(updated_state_2)
    if "error_logs" in update_3 and update_3["error_logs"]:
        print(f"❌ Respondent Agent 报错: {update_3['error_logs']}")
        return
    final_state = {**updated_state_2, **update_3}
    
    seeds = final_state.get("seed_responses", [])
    if len(seeds) != len(personas):
        print(f"⚠️ 警告: 画像数量 ({len(personas)}) 与种子回答数量 ({len(seeds)}) 不一致")
    else:
        print(f"✅ Respondent Agent 运行成功，生成了 {len(seeds)} 条种子回答")
    
    # 6. 执行流水线 - 第四步：Data Expansion Agent (数据扩增)
    print("\n--- 步骤 4: 运行 Data Expansion Agent (数据扩增) ---")
    update_4 = data_expansion_node(final_state)
    if "error_logs" in update_4 and update_4["error_logs"]:
        print(f"❌ Data Expansion Agent 报错: {update_4['error_logs']}")
        return
    final_state = {**final_state, **update_4}
    
    raw_data_path = final_state.get("raw_data_path")
    if os.path.exists(raw_data_path):
        print(f"✅ Data Expansion Agent 运行成功，原始数据已生成: {raw_data_path}")
    else:
        print("❌ Data Expansion Agent 运行失败，未找到生成的文件")
        return

    # 7. 执行流水线 - 第五步：Excel Conversion (格式化输出)
    print("\n--- 步骤 5: 运行 Excel Conversion (格式化输出) ---")
    output_xlsx_path = "data/output/final_survey_results.xlsx"
    q_file = "data/intermediate/questionnaire.json"
    
    success = convert_csv_to_xlsx(raw_data_path, q_file, output_xlsx_path)
    if success:
        print(f"✅ Excel 转换成功: {output_xlsx_path}")
    else:
        print("❌ Excel 转换失败")
        return

    # 8. 验证与输出报告
    print("\n" + "="*30 + " 最终验证报告 " + "="*30)
    
    # 打印问卷标题
    questionnaire = final_state.get("questionnaire", {})
    print(f"\n【问卷标题】: {questionnaire.get('survey_title', '未命名问卷')}")
    
    # 打印画像与回答摘要
    print(f"\n【画像与回答匹配情况】:")
    for i, p in enumerate(personas, 1):
        name = p.get("name_tag", "未知")
        proportion = p.get("proportion", 0)
        
        # 查找对应的回答
        matching_seed = next((s for s in seeds if s.get("persona_name") == name), None)
        status = "✅ 已回答" if matching_seed else "❌ 未回答"
        res_count = len(matching_seed.get("responses", {})) if matching_seed else 0
        
        print(f"  {i}. {name} (权重: {proportion:.2%}) - [{status}] ({res_count} 个字段)")

    # 统计验证
    total_proportion = sum(p.get("proportion", 0) for p in personas)
    print(f"\n【统计验证】:")
    print(f"  所有画像占比总和: {total_proportion:.4f}")
    if abs(total_proportion - 1.0) < 0.001:
        print("  ✅ 权重归一化验证通过")
    else:
        print(f"  ❌ 权重归一化验证失败")
        
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path)
        print(f"  ✅ 扩增数据条数验证: {len(df)} 条")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_workflow_6_personas()
