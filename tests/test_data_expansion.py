import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.state import SystemState
from src.agents.data_expansion_agent import data_expansion_node

def test_data_expansion():
    # 1. 加载画像数据
    p_file = "data/intermediate/personas.json"
    if not os.path.exists(p_file):
        print(f"❌ 找不到画像文件: {p_file}，请先运行 workflow 测试生成画像。")
        return
        
    with open(p_file, 'r', encoding='utf-8') as f:
        persona_data = json.load(f)
        personas = persona_data.get("personas", [])
        
    # 2. 初始化模拟状态
    state: SystemState = {
        "personas": personas,
        "raw_data_path": "",
        "current_step": "start",
        "error_logs": []
    }
    
    # 3. 运行数据扩增节点
    print("\n>>> 开始测试数据扩增节点...")
    update = data_expansion_node(state)
    
    if "error_logs" in update and update["error_logs"]:
        print(f"❌ 数据扩增节点报错: {update['error_logs']}")
        return
        
    raw_data_path = update.get("raw_data_path")
    print(f"✅ 节点运行成功，输出路径: {raw_data_path}")
    
    # 4. 验证生成的 CSV
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path)
        print(f"\n【验证报告】:")
        print(f"  - 样本总数: {len(df)} (预期约 5000)")
        print(f"  - 列名: {list(df.columns)}")
        print(f"  - 画像分布统计:")
        print(df['persona_name'].value_counts(normalize=True))
        
        # 验证数值范围 (假设 l1 是李克特量表题)
        likert_cols = [c for c in df.columns if c.startswith('l')]
        if likert_cols:
            print(f"\n【数值范围验证 ({likert_cols[0]})】:")
            print(df[likert_cols[0]].value_counts().sort_index())
            
            # 检查是否有超出 1-5 范围的值
            invalid = df[(df[likert_cols[0]] < 1) | (df[likert_cols[0]] > 5)]
            if invalid.empty:
                print(f"  ✅ {likert_cols[0]} 所有数值均在 1-5 范围内")
            else:
                print(f"  ❌ 发现 {len(invalid)} 条非法数值")
    else:
        print(f"❌ 找不到生成的 CSV 文件: {raw_data_path}")

if __name__ == "__main__":
    test_data_expansion()
