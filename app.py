import os
import json
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Project imports
from src.workflow.state import SystemState
from src.agents.survey_agent import survey_node
from src.agents.persona_agent import persona_node
from src.agents.respondent_agent import respondent_node
from src.agents.data_expansion_agent import data_expansion_node
from src.agents.open_ended_agent import open_ended_node
from src.agents.analysis_agent import analysis_agent_node
from src.agents.plotting_agent import plotting_agent_node
from src.agents.writer_agent import writer_agent_node

# 加载环境变量
load_dotenv()

def create_workflow():
    """创建并编译 LangGraph 工作流"""
    
    # 1. 初始化状态图
    workflow = StateGraph(SystemState)
    
    # 2. 添加所有节点
    workflow.add_node("survey_agent", survey_node)
    workflow.add_node("persona_agent", persona_node)
    workflow.add_node("respondent_agent", respondent_node)
    workflow.add_node("data_expansion_agent", data_expansion_node)
    workflow.add_node("open_ended_agent", open_ended_node)
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("plotting_agent", plotting_agent_node)
    workflow.add_node("writer_agent", writer_agent_node)
    
    # 3. 设置边和流程
    workflow.add_edge(START, "survey_agent")
    workflow.add_edge("survey_agent", "persona_agent")
    workflow.add_edge("persona_agent", "respondent_agent")
    
    # 并行分支 (Fork)
    workflow.add_edge("respondent_agent", "data_expansion_agent")
    workflow.add_edge("respondent_agent", "open_ended_agent")
    
    # 汇聚分支 (Join) -> 进入分析节点
    workflow.add_edge("data_expansion_agent", "analysis_agent")
    workflow.add_edge("open_ended_agent", "analysis_agent")
    
    # 最终顺序流
    workflow.add_edge("analysis_agent", "plotting_agent")
    workflow.add_edge("plotting_agent", "writer_agent")
    workflow.add_edge("writer_agent", END)
    
    # 4. 编译
    return workflow.compile()

def main():
    print("\n" + "🚀" * 30)
    print("      PaperAnvil: 全自动 AI 学术调研报告生成系统")
    print("🚀" * 30 + "\n")
    
    # 初始化状态
    initial_state: SystemState = {
        "topic": "企业内部用户中心模块的移动端适配体验调研",
        "persona_count": 5, # 生成 5 个核心画像
        "questionnaire": {},
        "personas": [],
        "seed_responses": [],
        "raw_data_path": "",
        "open_ended_detailed_responses": [],
        "analysis_insights": {},
        "thesis_draft": "",
        "plot_image_paths": [],
        "image_insertion_guide": [],
        "basic_stats": {},
        "semantic_stats": {},
        "error_logs": [],
        "current_step": "START"
    }
    
    # 创建工作流
    app = create_workflow()
    
    print("正在运行工作流，请稍候...")
    
    try:
        # 运行
        final_state = app.invoke(initial_state)
        
        print("\n" + "✅" * 30)
        print("      全流程生产完成！")
        print("✅" * 30 + "\n")
        
        if final_state.get("thesis_draft"):
            print(f"🎉 最终调研报告已生成：data/output/thesis_draft.md")
            print(f"📄 报告字符数：{len(final_state['thesis_draft'])}")
        
        if final_state.get("plot_image_paths"):
            print(f"📊 已生成图表数量：{len(final_state['plot_image_paths'])}")
            
    except Exception as e:
        print(f"\n❌ 工作流运行发生异常：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
