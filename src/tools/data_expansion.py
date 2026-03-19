import pandas as pd
import numpy as np
from typing import Dict, Any
import json

def expand_data(persona_json_path: str, output_csv_path: str, num_samples: int = 5000) -> Dict[str, Any]:
    """
    读取用户画像比例，注入噪声并批量扩增数据。
    
    Args:
        persona_json_path: 用户画像 JSON 文件路径
        output_csv_path: 输出 CSV 文件路径
        num_samples: 生成的样本数量，默认为 5000
        
    Returns:
        包含生成结果的字典，格式为 {
            "output_path": str,  # 生成的 CSV 文件路径
            "num_samples": int,  # 实际生成的样本数
            "success": bool      # 是否成功
        }
        
    Raises:
        FileNotFoundError: 如果画像文件不存在
        ValueError: 如果 JSON 格式不正确
    """
    try:
        with open(persona_json_path, 'r', encoding='utf-8') as f:
            personas = json.load(f)
        
        data = []
        for persona in personas:
            proportion = persona.get('proportion', 0.1)
            count = int(num_samples * proportion)
            
            for _ in range(count):
                row = {}
                for key, value in persona.items():
                    if key != 'proportion':
                        if isinstance(value, (int, float)):
                            noise = np.random.normal(0, 0.1 * abs(value))
                            row[key] = max(0, value + noise)
                        else:
                            row[key] = value
                data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        
        return {
            "output_path": output_csv_path,
            "num_samples": len(df),
            "success": True
        }
        
    except FileNotFoundError as e:
        return {
            "output_path": "",
            "num_samples": 0,
            "success": False,
            "error": f"文件未找到: {str(e)}"
        }
    except json.JSONDecodeError as e:
        return {
            "output_path": "",
            "num_samples": 0,
            "success": False,
            "error": f"JSON 解析错误: {str(e)}"
        }
    except Exception as e:
        return {
            "output_path": "",
            "num_samples": 0,
            "success": False,
            "error": f"未知错误: {str(e)}"
        }
