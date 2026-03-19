import pytest
import pandas as pd
import numpy as np
import json
import os
import sys

# 将项目根目录添加到 sys.path 以便导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.basic_stats_tool import BasicStatsTool

@pytest.fixture
def mock_data(tmp_path):
    questionnaire = {
        "demographics": [
            {"id": "d1", "question": "年龄"}
        ],
        "likert_scales": [
            {"id": "l1", "question": "满意度"}
        ]
    }
    
    df = pd.DataFrame({
        "d1": ["20-30"] * 60 + ["30-40"] * 40,
        "l1": [5] * 20 + [4] * 30 + [3] * 30 + [2] * 10 + [1] * 10
    })
    
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path), questionnaire

def test_basic_stats_tool(mock_data, tmp_path):
    csv_path, questionnaire = mock_data
    
    tool = BasicStatsTool(csv_path, questionnaire)
    results = tool.run_all()
    
    # 验证人口学特征
    assert "demographics_distribution" in results
    assert results["demographics_distribution"]["d1"]["counts"]["20-30"] == 60
    assert results["demographics_distribution"]["d1"]["percentages"]["20-30"] == 60.0
    
    # 验证李克特量表
    assert "likert_stats" in results
    assert results["likert_stats"]["l1"]["mean"] == pytest.approx(3.4)
    assert results["likert_stats"]["l1"]["distribution"]["5"] == 20
    
    # 验证保存
    output_dir = tmp_path / "output"
    tool.save_results(str(output_dir))
    assert os.path.exists(output_dir / "basic_stats.json")

if __name__ == "__main__":
    pytest.main([__file__])
