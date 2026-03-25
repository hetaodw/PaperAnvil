import os
import json
import pandas as pd
from typing import Dict, Any
from src.workflow.state import SystemState

def exporter_node(state: SystemState) -> dict:
    """
    数据导出节点。将模拟生成的画像数据和响应数据导出为 XLSX 格式。
    """
    try:
        print("\n>>> 正在准备数据导出 (JSON -> XLSX)...")
        
        # 1. 加载调研样本数据
        # 假设 seed_responses.json 存储了主要数据，或者从 state['seed_responses'] 获取
        responses = state.get("seed_responses", [])
        if not responses:
            # 尝试从本地加载（兜底）
            responses_path = "data/intermediate/seed_responses.json"
            if os.path.exists(responses_path):
                with open(responses_path, 'r', encoding='utf-8') as f:
                    responses = json.load(f)
        
        # 2. 加载画像数据以补全信息 (可选)
        personas = state.get("personas", [])
        persona_map = {p['name_tag']: p for p in personas}
        
        # 3. 构造 DataFrame
        rows = []
        for resp in responses:
            p_name = resp.get("persona_name")
            p_info = persona_map.get(p_name, {})
            
            row = {
                "Persona Name": p_name,
                "Gender": p_info.get("gender"),
                "Age": p_info.get("age"),
                "Job": p_info.get("job"),
                "Location": p_info.get("location")
            }
            # 合并问卷回答
            res_data = resp.get("responses", {})
            row.update(res_data)
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # 4. 导出
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)
        xlsx_path = os.path.join(output_dir, "research_data.xlsx")
        
        df.to_excel(xlsx_path, index=False)
        
        print(f"✅ 数据导出成功！Excel 文件已保存至: {xlsx_path}")
        
        return {
            "current_step": "exporter_agent"
        }
        
    except Exception as e:
        error_msg = f"[Exporter Agent] 运行失败: {str(e)}"
        print(error_msg)
        return {"error_logs": [error_msg], "current_step": "error"}
