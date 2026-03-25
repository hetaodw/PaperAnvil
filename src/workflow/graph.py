from langgraph.graph import StateGraph, END
from src.workflow.state import SystemState
from src.agents.survey_agent import survey_node as survey_agent_node
from src.agents.survey_ui_agent import survey_ui_node as survey_ui_agent_node
from src.agents.persona_agent import persona_agent_node
from src.agents.analysis_agent import analysis_agent_node
from src.agents.plotting_agent import plotting_agent_node
from src.agents.writer_agent import writer_agent_node
from src.agents.open_ended_agent import open_ended_node
from src.tools.data_expansion import expand_data
from src.tools.csv_to_xlsx import convert_csv_to_xlsx
from src.tools.csv_validator import validate_and_prepare_csv

def data_expansion_node(state: SystemState) -> SystemState:
    persona_json_path = "data/intermediate/personas.json"
    output_csv_path = "data/raw_data/simulated_data.csv"

    result = expand_data(persona_json_path, output_csv_path, num_samples=5000)

    return {
        **state,
        "raw_data_path": result["output_path"],
        "current_step": "data_expansion"
    }

def csv_validator_node(state: SystemState) -> SystemState:
    """
    验证和预处理用户提供的 CSV 文件。
    当 use_existing_csv=True 时，跳过数据生成，直接使用用户的数据。
    """
    csv_path = state.get("existing_csv_path", "")
    questionnaire = state.get("questionnaire", {})
    
    result = validate_and_prepare_csv(csv_path, questionnaire)
    
    if result["success"]:
        return {
            **state,
            "raw_data_path": result["output_path"],
            "current_step": "csv_validator"
        }
    else:
        error_msg = f"[CSV Validator] {result.get('error', '未知错误')}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "error_logs": [error_msg],
            "current_step": "error"
        }

def csv_to_xlsx_node(state: SystemState) -> SystemState:
    csv_path = state.get("raw_data_path", "data/raw_data/simulated_data.csv")
    questionnaire_path = "data/intermediate/questionnaire.json"
    output_xlsx_path = "data/output/final_survey_results.xlsx"

    convert_csv_to_xlsx(csv_path, questionnaire_path, output_xlsx_path)

    return {
        **state,
        "current_step": "csv_to_xlsx"
    }

def create_graph():
    workflow = StateGraph(SystemState)
    
    workflow.add_node("survey_agent", survey_agent_node)
    workflow.add_node("survey_ui_agent", survey_ui_agent_node)
    workflow.add_node("persona_agent", persona_agent_node)
    workflow.add_node("open_ended_agent", open_ended_node)
    workflow.add_node("data_expansion", data_expansion_node)
    workflow.add_node("csv_validator", csv_validator_node)
    workflow.add_node("csv_to_xlsx", csv_to_xlsx_node)
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("plotting_agent", plotting_agent_node)
    workflow.add_node("writer_agent", writer_agent_node)
    
    workflow.set_entry_point("survey_agent")
    
    workflow.add_edge("survey_agent", "survey_ui_agent")
    workflow.add_edge("survey_ui_agent", "persona_agent")
    
    # Fork after persona_agent
    # 根据是否使用已有 CSV，选择不同的路径
    # 如果 use_existing_csv=True，使用 csv_validator；否则使用 data_expansion
    workflow.add_edge("persona_agent", "open_ended_agent")
    workflow.add_edge("data_expansion", "analysis_agent")
    workflow.add_edge("csv_validator", "analysis_agent")
    workflow.add_edge("open_ended_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "plotting_agent")
    workflow.add_edge("plotting_agent", "writer_agent")
    workflow.add_edge("writer_agent", "csv_to_xlsx")
    workflow.add_edge("csv_to_xlsx", END)
    
    return workflow.compile()

def run_workflow_with_csv_check(initial_state):
    """
    运行工作流，根据 use_existing_csv 标志选择数据源。
    """
    from src.workflow.graph import create_graph
    
    app = create_graph()
    
    # 如果使用已有 CSV，直接从 csv_validator 开始，跳过 data_expansion
    if initial_state.get("use_existing_csv", False):
        print(">>> 检测到使用已有 CSV 文件，跳过数据生成步骤")
        # 修改工作流，直接从 csv_validator 开始
        # 这里需要手动处理状态流转
        from src.tools.csv_validator import validate_and_prepare_csv
        from src.agents.open_ended_agent import open_ended_node
        
        # 先运行 csv_validator
        result = validate_and_prepare_csv(
            initial_state.get("existing_csv_path", ""),
            initial_state.get("questionnaire", {})
        )
        
        if result["success"]:
            initial_state["raw_data_path"] = result["output_path"]
            print(f"✅ CSV 文件验证成功: {result['output_path']}")
        else:
            error_msg = f"[CSV Validator] {result.get('error', '未知错误')}"
            print(f"❌ {error_msg}")
            initial_state["error_logs"] = [error_msg]
            initial_state["current_step"] = "error"
            return initial_state
        
        # 然后运行 open_ended_agent
        open_ended_result = open_ended_node(initial_state)
        initial_state.update(open_ended_result)
        
        # 最后运行 analysis_agent 及后续节点
        from src.agents.analysis_agent import analysis_agent_node
        from src.agents.plotting_agent import plotting_agent_node
        from src.agents.writer_agent import writer_agent_node
        from src.tools.csv_to_xlsx import convert_csv_to_xlsx
        
        # analysis_agent
        analysis_result = analysis_agent_node(initial_state)
        initial_state.update(analysis_result)
        
        # plotting_agent
        plotting_result = plotting_agent_node(initial_state)
        initial_state.update(plotting_result)
        
        # writer_agent
        writer_result = writer_agent_node(initial_state)
        initial_state.update(writer_result)
        
        # csv_to_xlsx
        csv_path = initial_state.get("raw_data_path", "data/raw_data/simulated_data.csv")
        questionnaire_path = "data/intermediate/questionnaire.json"
        output_xlsx_path = "data/output/final_survey_results.xlsx"
        convert_csv_to_xlsx(csv_path, questionnaire_path, output_xlsx_path)
        initial_state["current_step"] = "csv_to_xlsx"
        
        return initial_state
    else:
        # 正常流程，使用 data_expansion
        return app.invoke(initial_state)
