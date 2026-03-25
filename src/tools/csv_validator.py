import pandas as pd
import numpy as np
import os
from typing import Dict, Any

def validate_and_prepare_csv(csv_path: str, questionnaire: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证并预处理用户提供的 CSV 文件。
    
    处理步骤：
    1. 检查文件是否存在
    2. 读取 CSV 并检查格式
    3. 处理 NaN 值
    4. 验证列名与问卷匹配
    5. 保存处理后的文件
    
    Args:
        csv_path: 用户提供的 CSV 文件路径
        questionnaire: 问卷结构字典
        
    Returns:
        结果字典，包含 success, output_path, num_samples, error 等
    """
    try:
        print(f"\n>>> 开始验证和预处理 CSV 文件: {csv_path}")
        
        # 1. 检查文件是否存在
        if not os.path.exists(csv_path):
            return {
                "success": False,
                "error": f"文件不存在: {csv_path}"
            }
        
        # 2. 读取 CSV
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"  - 原始数据行数: {len(df)}, 列数: {len(df.columns)}")
        
        # 3. 处理 NaN 值
        if df.isnull().any().any():
            print("  - 检测到 NaN 值，正在处理...")
            
            # 数值列用均值填充
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    print(f"    - 数值列 {col}: 用均值 {mean_val:.2f} 填充 {df[col].isnull().sum()} 个空值")
            
            # 字符串列用 "未知" 填充
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna("未知")
                    print(f"    - 字符串列 {col}: 用 '未知' 填充 {df[col].isnull().sum()} 个空值")
        
        # 4. 验证列名
        expected_likert_cols = [q["id"] for q in questionnaire.get("likert_scales", [])]
        expected_demo_cols = [q["id"] for q in questionnaire.get("demographics", [])]
        expected_cols = set(expected_likert_cols + expected_demo_cols)
        
        actual_cols = set(df.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols - {"persona_name"}
        
        if missing_cols:
            print(f"  - ⚠️ 缺失列: {missing_cols}")
            # 填充缺失的李克特题（用中位数 3）
            for col in missing_cols:
                if col in expected_likert_cols:
                    df[col] = 3
                    print(f"    - 缺失李克特列 {col}: 填充默认值 3")
                elif col in expected_demo_cols:
                    df[col] = "未知"
                    print(f"    - 缺失人口统计列 {col}: 填充默认值 '未知'")
        
        if extra_cols:
            print(f"  - ℹ️ 额外列 (将被忽略): {extra_cols}")
        
        # 5. 保存处理后的文件
        output_dir = "data/raw_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "simulated_data.csv")
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ CSV 验证和预处理完成，已保存至: {output_path}")
        print(f"   最终数据: {len(df)} 行, {len(df.columns)} 列")
        
        return {
            "success": True,
            "output_path": output_path,
            "num_samples": len(df),
            "num_columns": len(df.columns),
            "missing_cols": list(missing_cols),
            "extra_cols": list(extra_cols)
        }
        
    except Exception as e:
        error_msg = f"CSV 验证和预处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

if __name__ == "__main__":
    # 测试
    import json
    with open("data/intermediate/questionnaire.json", "r", encoding="utf-8") as f:
        q = json.load(f)
    
    result = validate_and_prepare_csv("data/raw_data/test.csv", q)
    print(result)
