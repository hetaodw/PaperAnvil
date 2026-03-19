import pytest
import pandas as pd
import numpy as np
import json
import os
import sys

# 将项目根目录添加到 sys.path 以便导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.state_tool import StateTool

@pytest.fixture
def mock_data(tmp_path):
    # 创建模拟问卷
    questionnaire = {
        "demographics": [
            {"id": "d1", "question": "年龄"},
            {"id": "d2", "question": "性别"}
        ],
        "likert_scales": [
            {"id": "l1", "question": "满意度1"},
            {"id": "l2", "question": "满意度2"},
            {"id": "l3", "question": "总满意度"}
        ]
    }
    
    # 创建具有相关性的模拟数据
    n_samples = 200
    l1 = np.random.randint(1, 6, n_samples)
    # l2 与 l1 正相关
    l2 = np.clip(l1 + np.random.randint(-1, 2, n_samples), 1, 5)
    # l3 与 l1, l2 都相关
    l3 = np.clip((l1 + l2) // 2 + np.random.randint(0, 2, n_samples), 1, 5)
    
    df = pd.DataFrame({
        "persona_name": ["A"] * (n_samples // 2) + ["B"] * (n_samples // 2),
        "d1": np.random.choice(["20-30", "30-40"], n_samples),
        "d2": np.random.choice(["男", "女"], n_samples),
        "l1": l1,
        "l2": l2,
        "l3": l3,
    })
    
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path), questionnaire

def test_state_tool_analysis(mock_data, tmp_path):
    csv_path, questionnaire = mock_data
    
    tool = StateTool(csv_path, questionnaire)
    insights = tool.run_all()
    
    # 验证聚类
    assert "clustering" in insights
    assert "kmeans_centers" in insights["clustering"]
    
    # 验证异常检测
    assert "anomaly_detection" in insights
    assert insights["anomaly_detection"]["anomaly_count"] > 0
    
    # 验证相关性
    assert "correlation_analysis" in insights
    assert len(insights["correlation_analysis"]["top_correlations"]) > 0
    
    # 验证特征贡献
    assert "feature_contribution" in insights
    assert insights["feature_contribution"]["target_variable"] == "l3"
    
    # 验证保存
    output_dir = tmp_path / "output"
    tool.save_results(str(output_dir))
    assert os.path.exists(output_dir / "analysis_results.json")

def test_empty_analysis():
    # 测试边界情况
    pass # 可以在这里添加更多测试

if __name__ == "__main__":
    # 支持直接使用 python 运行
    pytest.main([__file__])
