import os
from dotenv import load_dotenv
from src.workflow.graph import create_graph

load_dotenv()

def main():
    graph = create_graph()
    
    thread_id = f"user_center_module_auth_analysis_{int(os.times()[4])}"
    
    initial_state = {
        "thread_id": thread_id,
        "topic": "用户中心模块认证流程分析",
        "raw_data_path": "",
        "plot_image_paths": [],
        "analysis_insights": {},
        "thesis_draft": "",
        "error_logs": [],
        "current_step": "start"
    }
    
    result = graph.invoke(initial_state)
    print(result)

if __name__ == "__main__":
    main()
