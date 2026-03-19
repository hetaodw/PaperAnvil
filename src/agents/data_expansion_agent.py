import os
from src.workflow.state import SystemState
from src.tools.data_expansion import expand_data

def data_expansion_node(state: SystemState) -> dict:
    """
    数据扩增节点。
    
    使用 src.tools.data_expansion 工具，根据画像比例和分布特征生成 5000 条模拟数据。
    
    Args:
        state: 全局状态字典，包含画像列表 personas
        
    Returns:
        更新后的状态字典，包含 raw_data_path
    """
    try:
        personas = state.get("personas", [])
        if not personas:
            error_msg = "[Data Expansion] 错误: 状态中没有画像数据，无法进行扩增。"
            print(error_msg)
            return {"error_logs": [error_msg], "current_step": "error"}
            
        # 设置输出路径
        raw_data_path = "data/raw_data/simulated_data.csv"
        
        # 默认生成 5000 条
        total_samples = 5000
        
        result = expand_data(personas, raw_data_path, total_samples=total_samples)
        
        if result["success"]:
            return {
                "raw_data_path": raw_data_path,
                "current_step": "data_expansion"
            }
        else:
            error_msg = f"[Data Expansion] 工具运行失败: {result.get('error', '未知错误')}"
            print(error_msg)
            return {"error_logs": [error_msg], "current_step": "error"}
            
    except Exception as e:
        error_msg = f"[Data Expansion] 运行发生异常: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
