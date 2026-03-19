import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any, List

def expand_data(personas: List[Dict[str, Any]], output_csv_path: str, total_samples: int = 5000) -> Dict[str, Any]:
    """
    根据画像分布、李克特正态分布和人口统计特征扩增模拟数据。
    
    Args:
        personas: 画像列表，每个画像需包含 proportion, demographics_fixed, likert_distribution
        output_csv_path: 输出 CSV 文件路径
        total_samples: 总样本数量，默认为 5000
        
    Returns:
        结果字典
    """
    try:
        all_data = []
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        print(f"\n>>> 开始数据扩增，目标总样本数: {total_samples}")
        
        for persona in personas:
            name = persona.get("name_tag", "未知")
            proportion = persona.get("proportion", 0.0)
            count = int(total_samples * proportion)
            
            if count <= 0:
                continue
                
            print(f"  - 正在为画像 [{name}] 生成 {count} 条样本 (占比: {proportion:.2%})")
            
            # 准备该画像的固定特征和分布特征
            demographics = persona.get("demographics_fixed", {})
            likert_dist = persona.get("likert_distribution", {})
            
            for _ in range(count):
                row = {
                    "persona_name": name,
                }
                
                # 1. 填入人口统计特征 (固定值)
                for q_id, val in demographics.items():
                    row[q_id] = val
                    
                # 2. 采样李克特量表 (正态分布)
                for q_id, dist in likert_dist.items():
                    mu = dist.get("mu", 3.0)
                    sigma = dist.get("sigma", 0.5)
                    
                    # 采样并取整、裁剪
                    sample = np.random.normal(mu, sigma)
                    score = int(round(sample))
                    score = max(1, min(5, score)) # 强制 1-5 范围
                    
                    row[q_id] = score
                
                all_data.append(row)
        
        # 转换为 DataFrame 并保存
        df = pd.DataFrame(all_data)
        
        # 随机打乱数据顺序，增加真实感
        df = df.sample(frac=1).reset_index(drop=True)
        
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 数据扩增完成，实际生成 {len(df)} 条样本，已保存至 {output_csv_path}")
        
        return {
            "output_path": output_csv_path,
            "num_samples": len(df),
            "success": True
        }
        
    except Exception as e:
        error_msg = f"数据扩增失败: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "output_path": "",
            "num_samples": 0,
            "success": False,
            "error": error_msg
        }

if __name__ == "__main__":
    # 简单测试逻辑
    test_personas = [
        {
            "name_tag": "测试画像A",
            "proportion": 0.6,
            "demographics_fixed": {"d1": "男", "d2": "20-30"},
            "likert_distribution": {"l1": {"mu": 4.5, "sigma": 0.3}, "l2": {"mu": 2.1, "sigma": 0.8}}
        },
        {
            "name_tag": "测试画像B",
            "proportion": 0.4,
            "demographics_fixed": {"d1": "女", "d2": "30-40"},
            "likert_distribution": {"l1": {"mu": 3.0, "sigma": 0.5}, "l2": {"mu": 3.5, "sigma": 0.4}}
        }
    ]
    expand_data(test_personas, "data/raw_data/test_expanded.csv", total_samples=100)
