import pytest
import os
import json
import sys
from unittest.mock import MagicMock, patch

# 将项目根目录添加到 sys.path 以便导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.open_ended_agent import open_ended_node

@pytest.fixture
def mock_state():
    return {
        "topic": "测试主题",
        "questionnaire": {
            "open_ended": [
                {"id": "o1", "question": "你对该产品的第一个印象是什么？"}
            ]
        },
        "personas": [
            {
                "name_tag": "测试用户-张三",
                "age": 25,
                "job": "学生",
                "personality": "积极开朗",
                "location": "北京",
                "proportion": 0.02  # 稀有画像，应生成更多份数
            }
        ],
        "current_step": "persona_agent"
    }

@patch("src.agents.open_ended_agent.OpenAI")
def test_open_ended_node_basic(mock_openai, mock_state):
    # 模拟 OpenAI 响应
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content='{"persona_name": "测试用户-张三", "responses": {"o1": "这是一个非常详细的回答，满足 150-250 字的要求，包含了具体的场景、痛点和情绪。"}}'))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    
    result = open_ended_node(mock_state)
    
    assert "open_ended_detailed_responses" in result
    # 因为 proportion 为 0.02 (< 0.05)，按照新逻辑，应该生成 2 份
    assert mock_client.chat.completions.create.call_count == 2
    assert len(result["open_ended_detailed_responses"]) == 2
    assert result["current_step"] == "open_ended_agent"
    
    # 检查文件是否生成
    assert os.path.exists("data/intermediate/open_ended_responses.json")

if __name__ == "__main__":
    pytest.main([__file__])
