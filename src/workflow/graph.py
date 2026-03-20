from langgraph.graph import StateGraph, END
from src.workflow.state import SystemState
from src.agents.survey_agent import survey_node as survey_agent_node
from src.agents.survey_ui_agent import survey_ui_node as survey_ui_agent_node
from src.agents.persona_agent import persona_node as persona_agent_node
from src.agents.analysis_agent import analysis_agent_node
from src.agents.plotting_agent import plotting_agent_node
from src.agents.writer_agent import writer_agent_node
from src.agents.open_ended_agent import open_ended_node
from src.tools.data_expansion import expand_data
from src.tools.csv_to_xlsx import convert_csv_to_xlsx

def data_expansion_node(state: SystemState) -> SystemState:
    persona_json_path = "data/intermediate/personas.json"
    output_csv_path = "data/raw_data/simulated_data.csv"

    result = expand_data(persona_json_path, output_csv_path, num_samples=5000)

    return {
        **state,
        "raw_data_path": result["output_path"],
        "current_step": "data_expansion"
    }

def csv_to_xlsx_node(state: SystemState) -> SystemState:
    csv_path = "data/raw_data/simulated_data.csv"
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
    workflow.add_node("csv_to_xlsx", csv_to_xlsx_node)
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("plotting_agent", plotting_agent_node)
    workflow.add_node("writer_agent", writer_agent_node)
    
    workflow.set_entry_point("survey_agent")
    
    workflow.add_edge("survey_agent", "survey_ui_agent")
    workflow.add_edge("survey_ui_agent", "persona_agent")
    
    # Fork after persona_agent
    workflow.add_edge("persona_agent", "data_expansion")
    workflow.add_edge("persona_agent", "open_ended_agent")
    
    # Currently both flow independently, but usually we need to wait for both.
    # For simplicity in this graph structure, we'll have them both lead to analysis.
    # Note: Analysis Agent should be able to handle partial state updates if running in parallel.
    workflow.add_edge("data_expansion", "analysis_agent")
    workflow.add_edge("open_ended_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "plotting_agent")
    workflow.add_edge("plotting_agent", "writer_agent")
    workflow.add_edge("writer_agent", "csv_to_xlsx")
    workflow.add_edge("csv_to_xlsx", END)
    
    return workflow.compile()
