import pandas as pd
import json
import os
from typing import Dict, Any

def convert_csv_to_xlsx(csv_path: str, questionnaire_path: str, output_xlsx_path: str) -> bool:
    """
    将 CSV 数据转换为用户友好的 XLSX 格式。
    1. 替换 ID 为真实题目文本。
    2. 移除 persona_name 辅助列。
    
    Args:
        csv_path: 原始 CSV 文件路径
        questionnaire_path: 问卷结构 JSON 路径
        output_xlsx_path: 输出 XLSX 路径
    """
    try:
        # 1. 加载数据
        if not os.path.exists(csv_path):
            print(f"❌ 错误: 找不到 CSV 文件 {csv_path}")
            return False
            
        if not os.path.exists(questionnaire_path):
            print(f"❌ 错误: 找不到问卷文件 {questionnaire_path}")
            return False
            
        df = pd.read_csv(csv_path)
        with open(questionnaire_path, 'r', encoding='utf-8') as f:
            questionnaire = json.load(f)
            
        # 2. 构建映射字典 {id: question_text}
        mapping = {}
        
        # 处理人口统计题
        for item in questionnaire.get("demographics", []):
            mapping[item["id"]] = item["question"]
            
        # 处理李克特量表题
        for item in questionnaire.get("likert_scales", []):
            mapping[item["id"]] = item["question"]
            
        # 3. 数据清洗
        # 移除不需要的列
        if "persona_name" in df.columns:
            df = df.drop(columns=["persona_name"])
            
        # 仅保留在映射表中的列（或者保留所有列但替换名称）
        new_columns = []
        for col in df.columns:
            if col in mapping:
                new_columns.append(mapping[col])
            else:
                new_columns.append(col)
        
        df.columns = new_columns
        
        # 4. 保存为 XLSX
        os.makedirs(os.path.dirname(output_xlsx_path), exist_ok=True)
        
        # 使用 ExcelWriter 可以进行一些简单的样式设置（可选）
        df.to_excel(output_xlsx_path, index=False, engine='openpyxl')
        
        print(f"✅ Excel 转换完成: {output_xlsx_path}")
        print(f"   - 已处理 {len(df)} 条数据")
        print(f"   - 已替换 {len(mapping)} 个表头 ID 为真实题目")
        
        return True
        
    except Exception as e:
        print(f"❌ Excel 转换失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 模拟运行
    convert_csv_to_xlsx(
        "data/raw_data/simulated_data.csv",
        "data/intermediate/questionnaire.json",
        "data/output/final_survey_results.xlsx"
    )
