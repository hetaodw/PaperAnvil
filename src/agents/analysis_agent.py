import os
import json
from src.workflow.state import SystemState
from src.tools.state_tool import StateTool
from src.tools.basic_stats_tool import BasicStatsTool
from src.tools.semantic_tool import SemanticTool

def analysis_agent_node(state: SystemState) -> dict:
    """
    全方位数据分析节点。
    
    1. 针对量化 CSV 数据：运行 BasicStatsTool 和 StateTool。
    2. 针对质性开放题数据：运行 SemanticTool。
    """
    print("\n" + "="*50)
    print("🔬 Analysis Agent: 开始多维度交叉分析...")
    print("="*50)
    
    insights = {}
    
    # --- 1. 定量分析 (Quantitative Analysis) ---
    raw_data_path = state.get("raw_data_path", "data/raw_data/simulated_data.csv")
    questionnaire = state.get("questionnaire", {})
    
    if os.path.exists(raw_data_path):
        print("\n>>> 正在进行定量分析 (Basic Stats & Machine Learning)...")
        # 基础统计
        bst = BasicStatsTool(raw_data_path, questionnaire)
        stats_results = bst.run_all()
        bst.save_results()
        insights["basic_stats"] = stats_results
        
        # 高级状态分析 (聚类、异常、相关性、特征重要性)
        st = StateTool(raw_data_path, questionnaire)
        advanced_results = st.run_all()
        st.save_results()
        insights["advanced_analysis"] = advanced_results
    else:
        print(f"⚠️ [Analysis Agent] 未找到量化数据文件: {raw_data_path}")

    # --- 2. 定性语义分析 (Qualitative Semantic Analysis) ---
    open_ended_path = "data/intermediate/open_ended_responses.json"
    if os.path.exists(open_ended_path):
        print("\n>>> 正在进行定性语义分析 (Topic Modeling, ABSA, Clustering)...")
        sem = SemanticTool(open_ended_path)
        semantic_results = sem.run_all()
        sem.save_results()
        insights["semantic_analysis"] = semantic_results
    else:
        print(f"⚠️ [Analysis Agent] 未找到开放题响应数据文件: {open_ended_path}")

    print("\n✅ 分析完成。")
    
    return {
        **state,
        "analysis_insights": insights,
        "current_step": "analysis_agent"
    }
