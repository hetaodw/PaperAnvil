import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.survey_agent import survey_node
from src.workflow.state import SystemState


def test_survey_agent():
    """
    测试问卷生成 Agent。
    
    测试主题：企业内部用户中心模块的移动端适配体验调研
    """
    print("=" * 60)
    print("开始测试问卷生成 Agent")
    print("=" * 60)
    
    initial_state: SystemState = {
        "thread_id": "test_survey_001",
        "topic": "企业内部用户中心模块的移动端适配体验调研",
        "questionnaire": {},
        "personas": [],
        "raw_data_path": "",
        "plot_image_paths": [],
        "analysis_insights": {},
        "thesis_draft": "",
        "error_logs": [],
        "current_step": "start"
    }
    
    print(f"\n调研主题：{initial_state['topic']}")
    print("\n正在调用问卷生成 Agent...\n")
    
    result = survey_node(initial_state)
    
    print("=" * 60)
    print("Agent 执行结果")
    print("=" * 60)
    
    if "error_logs" in result and result["error_logs"]:
        print("\n❌ 执行失败：")
        for error in result["error_logs"]:
            print(f"  - {error}")
        print(f"\n当前步骤：{result['current_step']}")
    else:
        print("\n✅ 执行成功！")
        print(f"\n当前步骤：{result['current_step']}")
        
        if "questionnaire" in result and result["questionnaire"]:
            print("\n生成的问卷结构：")
            print(json.dumps(result["questionnaire"], ensure_ascii=False, indent=2))
            
            print("\n问卷标题：", result["questionnaire"].get("survey_title", "未设置"))
            print("人口统计问题数量：", len(result["questionnaire"].get("demographics", [])))
            print("李克特量表题数量：", len(result["questionnaire"].get("likert_scales", [])))
            print("开放性问题数量：", len(result["questionnaire"].get("open_ended", [])))
        else:
            print("\n⚠️  未生成问卷结构")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    test_survey_agent()
