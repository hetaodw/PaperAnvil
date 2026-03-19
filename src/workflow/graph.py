from langgraph.graph import StateGraph, END
from src.workflow.state import SystemState
from src.agents.survey_agent import survey_agent_node
from src.agents.persona_agent import persona_agent_node
from src.agents.analysis_agent import analysis_agent_node
from src.agents.plotting_agent import plotting_agent_node
from src.agents.writer_agent import writer_agent_node
from src.tools.data_expansion import expand_data

def data_expansion_node(state: SystemState) -> SystemState:
    persona_json_path = "data/intermediate/personas.json"
    output_csv_path = "data/raw_data/simulated_data.csv"
    
    result = expand_data(persona_json_path, output_csv_path, num_samples=5000)
    
    return {
        **state,
        "raw_data_path": result["output_path"],
        "current_step": "data_expansion"
    }

def create_graph():
    workflow = StateGraph(SystemState)
    
    workflow.add_node("survey_agent", survey_agent_node)
    workflow.add_node("persona_agent", persona_agent_node)
    workflow.add_node("data_expansion", data_expansion_node)
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("plotting_agent", plotting_agent_node)
    workflow.add_node("writer_agent", writer_agent_node)
    
    workflow.set_entry_point("survey_agent")
    
    workflow.add_edge("survey_agent", "persona_agent")
    workflow.add_edge("persona_agent", "data_expansion")
    workflow.add_edge("data_expansion", "analysis_agent")
    workflow.add_edge("analysis_agent", "plotting_agent")
    workflow.add_edge("plotting_agent", "writer_agent")
    workflow.add_edge("writer_agent", END)
    
    return workflow.compile()
