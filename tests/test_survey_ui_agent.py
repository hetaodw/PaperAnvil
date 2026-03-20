import os
import sys
import json

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.survey_ui_agent import survey_ui_node
from src.workflow.state import SystemState

def test_survey_ui_agent():
    # 模拟问卷结构
    mock_questionnaire = {
        "survey_title": "泰安市留学生视角下的中国医疗体系发展调研",
        "demographics": [
            {"id": "d1", "question": "您的性别是？", "options": ["男", "女"]},
            {"id": "d2", "question": "您的年龄段是？", "options": ["18-25", "26-35", "36-50", "50+"]}
        ],
        "likert_scales": [
            {"id": "l1", "question": "您对泰安市公立医院的服务态度是否满意？", "scale_range": [1, 5], "labels": {"1": "极不满意", "5": "极其满意"}},
            {"id": "l2", "question": "您认为医院的预约挂号流程是否便捷？", "scale_range": [1, 5], "labels": {"1": "极不便捷", "5": "极其便捷"}}
        ],
        "open_ended": [
            {"id": "o1", "question": "您对中国医疗体系最深刻的印象是什么？"},
            {"id": "o2", "question": "您有什么具体的改进建议吗？"}
        ]
    }
    
    state: SystemState = {
        "topic": "泰安市留学生视角下的中国医疗体系发展",
        "questionnaire": mock_questionnaire,
        "persona_count": 3,
        "personas": [],
        "seed_responses": [],
        "raw_data_path": "",
        "plot_image_paths": [],
        "analysis_insights": {},
        "image_insertion_guide": [],
        "open_ended_detailed_responses": [],
        "thesis_draft": "",
        "basic_stats": {},
        "semantic_stats": {},
        "error_logs": [],
        "current_step": "start"
    }
    
    # 强制覆盖环境变量中的 MODEL_NAME 用于测试
    os.environ["MODEL_NAME"] = "qwen-plus"
    
    print(">>> 开始测试 Survey UI Agent (Model: qwen-plus)...")
    result = survey_ui_node(state)
    
    if "error_logs" in result and result["error_logs"]:
        print(f"❌ 测试失败: {result['error_logs']}")
    else:
        print("✅ 测试完成！请检查 data/output/survey.html 文件。")
        output_path = os.path.join("data", "output", "survey.html")
        if os.path.exists(output_path):
            print(f"文件大小: {os.path.getsize(output_path)} 字节")

if __name__ == "__main__":
    test_survey_ui_agent()
