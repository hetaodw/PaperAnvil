import os
import json
import sys
import pytest
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.open_ended_agent import open_ended_node
from src.agents.analysis_agent import analysis_agent_node
from src.workflow.state import SystemState

load_dotenv()

def test_workflow_fork_full_mock():
    """
    全 Mock 版本，不依赖外部 API，验证 Fork 和 Analysis 的逻辑链路。
    """
    print("\n" + "="*60)
    print("🚀 开始全 Mock 版 Fork 流程深度验证...")
    print("="*60)
    
    # 1. 构造 Mock 画像数据
    mock_questionnaire = {
        "survey_title": "智慧办公系统满意度调研",
        "demographics": [{"id": "d1", "question": "您的职业？", "options": ["开发", "产品", "设计", "运营"]}],
        "likert_scales": [{"id": "l1", "question": "系统响应速度", "scale_range": [1, 5], "labels": {"1": "极慢", "5": "极快"}}],
        "open_ended": [{"id": "o1", "question": "您认为系统在多设备同步上最大的痛点是什么？"}]
    }
    
    mock_personas = [
        {
            "name_tag": "极简主义-王工",
            "age": 32,
            "job": "后端架构师",
            "proportion": 0.4,
            "demographics_fixed": {"d1": "开发"},
            "likert_distribution": {"l1": {"mu": 4.5, "sigma": 0.3}}
        },
        {
            "name_tag": "感性派-小美",
            "age": 24,
            "job": "UI设计师",
            "proportion": 0.04,
            "demographics_fixed": {"d1": "设计"},
            "likert_distribution": {"l1": {"mu": 2.5, "sigma": 0.8}}
        }
    ]
    
    state: SystemState = {
        "thread_id": "test_fork_full_mock",
        "topic": "智慧办公系统调研",
        "questionnaire": mock_questionnaire,
        "personas": mock_personas,
        "seed_responses": [],
        "raw_data_path": "",
        "open_ended_detailed_responses": [],
        "analysis_insights": {},
        "thesis_draft": "",
        "plot_image_paths": [],
        "error_logs": [],
        "current_step": "persona_agent"
    }

    # 2. Step 3a: Data Expansion (Local logic)
    print("\n[Step 3a] Data Expansion...")
    from src.tools.data_expansion import expand_data
    result = expand_data(mock_personas, "data/raw_data/simulated_data.csv", total_samples=50)
    state["raw_data_path"] = result["output_path"]

    # 3. Step 3b: OpenEnded Agent (Mocked LLM)
    print("\n[Step 3b] OpenEnded Agent (Mocked)...")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    # 注意：agent 期望的是 JSON 字符串
    # 我们 Mock 不同的内容以避免 CountVectorizer 报错
    mock_responses = [
        json.dumps({
            "persona_name": "极简主义-王工",
            "responses": {"o1": "系统同步延迟是最大的设计问题，尤其在多设备切换时。"}
        }),
        json.dumps({
            "persona_name": "极简主义-王工",
            "responses": {"o1": "我非常在意效率，但现在的加载速度太慢了，影响工作心情。"}
        }),
        json.dumps({
            "persona_name": "感性派-小美",
            "responses": {"o1": "界面很好看，但是有些按钮太小了，经常点错，很烦人。"}
        })
    ]
    
    mock_counter = [0]
    def side_effect(*args, **kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = mock_responses[mock_counter[0] % len(mock_responses)]
        mock_counter[0] += 1
        return resp

    with patch("openai.resources.chat.completions.Completions.create", side_effect=side_effect):
        state.update(open_ended_node(state))
    
    assert len(state["open_ended_detailed_responses"]) > 0
    print(f"  ✅ 已生成 {len(state['open_ended_detailed_responses'])} 条模拟深度回答。")

    # 4. Step 4: Analysis Agent (Mocked DashScope & OpenAI)
    print("\n[Step 4] Analysis Agent (Mocked Tools)...")
    
    # Mock DashScope Embedding
    def ds_side_effect(model, input):
        m = MagicMock()
        m.status_code = 200
        # 返回与 input 长度一致的 embeddings
        m.output = {'embeddings': [{'embedding': [0.1] * 1536} for _ in range(len(input))]}
        return m
    
    # Mock OpenAI for ABSA/Topics in tools
    mock_tool_llm_resp = MagicMock()
    mock_tool_llm_resp.choices = [MagicMock()]
    mock_tool_llm_resp.choices[0].message.content = json.dumps({
        "topics": [{"topic": "同步延迟", "keywords": ["延迟", "登录"], "count": 5}],
        "sentiments": [{"aspect": "多端同步", "sentiment": "中性", "reason": "延迟问题"}]
    })

    with patch("dashscope.TextEmbedding.call", side_effect=ds_side_effect), \
         patch("openai.resources.chat.completions.Completions.create", return_value=mock_tool_llm_resp):
        state.update(analysis_agent_node(state))

    # 5. 最终验证
    print("\n🔍 流程产出深度验证...")
    # 产出文件检查
    files = [
        "data/raw_data/simulated_data.csv",
        "data/intermediate/open_ended_responses.json",
        "data/intermediate/basic_stats.json",
        "data/intermediate/analysis_results.json",
        "data/intermediate/semantic_analysis.json"
    ]
    for f in files:
        assert os.path.exists(f) and os.path.getsize(f) > 0
        print(f"  ✅ 文件存在: {f}")

    print("\n✅ 全路经集成测试通过！(定量 + 定性并行 Fork 逻辑验证成功)")

if __name__ == "__main__":
    test_workflow_fork_full_mock()
