import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.workflow.state import SystemState

# 导入实际使用的分析工具
from src.tools.basic_stats_tool import BasicStatsTool
from src.tools.state_tool import StateTool
from src.tools.semantic_tool import SemanticTool

load_dotenv()

def analysis_agent_node(state: SystemState) -> dict:
    """
    数据分析节点 (双阶段架构：原子化工具整合 -> LLM 深度详析与绘图规划)
    """
    print("\n" + "="*60)
    print("🚀 [Analysis Agent] 开始执行多维数据分析 (定量+定性)...")
    print("="*60)

    try:
        raw_data_path = state.get("raw_data_path")
        questionnaire = state.get("questionnaire")
        
        # 针对语义分析，尝试从状态或默认路径获取 open_ended_responses.json
        # 在 test_workflow_fork.py 中，该文件路径是固定的
        responses_path = "data/intermediate/open_ended_responses.json"
        
        if not raw_data_path or not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"找不到原始数据文件: {raw_data_path}")
        if not questionnaire:
            raise ValueError("State 中缺少问卷 (questionnaire) 信息")

        # ---------------------------------------------------------
        # 步骤 0: 调用原子化分析工具 (定量 + 定性)
        # ---------------------------------------------------------
        print(">>> 步骤 0: 正在执行底层统计学与机器学习算法工具...")
        
        # 1. 基础描述性统计
        bst = BasicStatsTool(raw_data_path, questionnaire)
        basic_stats = bst.run_all()
        bst.save_results()
        
        # 2. 高级机器学习分析 (聚类、异常、相关性)
        st = StateTool(raw_data_path, questionnaire)
        advanced_stats = st.run_all()
        st.save_results()
        
        # 3. 语义分析 (针对开放题)
        semantic_stats = {}
        if os.path.exists(responses_path):
            sem_tool = SemanticTool(responses_path)
            semantic_stats = sem_tool.run_all()
            sem_tool.save_results()
        else:
            print(f"⚠️ 未找到开放题回答文件: {responses_path}，将跳过语义分析。")

        # 整合所有原始统计结果发送给 LLM
        raw_stats_context = {
            "basic_statistics": basic_stats,
            "machine_learning_insights": advanced_stats,
            "semantic_qualitative_insights": semantic_stats
        }
        
        # 初始化大模型客户端
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY")),
            base_url=os.getenv("DASHSCOPE_BASE_URL", os.getenv("OPENAI_BASE_URL")),
        )
        model_name = os.getenv("MODEL_NAME", "gpt-4o")

        # ---------------------------------------------------------
        # 步骤 1: 洞察筛选 (过滤噪声，提取高价值商业/学术发现)
        # ---------------------------------------------------------
        print(">>> 步骤 1: 正在进行洞察提取 (从海量统计结果中过滤 Top 价值特征)...")
        
        filter_prompt = f"""你是一个资深的数据科学家。
下面是针对调研数据运行多种算法（基础统计、聚类、异常检测、相关性、LDA主题建模、语义聚类）后得出的综合分析结果。
由于数据量较大，其中包含部分常识性结论。

【综合统计分析结果摘要】
{json.dumps(raw_stats_context, ensure_ascii=False)[:10000]} # 防止 Token 超限

【你的任务】
请识别出 3-5 个最具“商业价值”和“学术研究价值”的核心发现。
特别关注：
1. 定量数据中的显著相关或异常簇。
2. 定性数据（语义分析）中频率最高或情感最强烈的反馈。
3. 定量与定性结论的交叉验证。

只输出合法的 JSON 列表，格式如下，不要包含 Markdown 包裹符：
[
  {{"finding_id": "f1", "observation": "发现描述", "why_it_matters": "价值说明"}}
]"""

        response_1 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": filter_prompt}],
            temperature=0.3
        )
        
        high_value_findings_str = response_1.choices[0].message.content.strip()
        print("✅ 洞察筛选完成。")

        # ---------------------------------------------------------
        # 步骤 2: 详细解读与图表规划
        # ---------------------------------------------------------
        print(">>> 步骤 2: 正在进行深度报告撰写规划与绘图指令生成...")
        
        detailed_prompt = f"""你是一位首席分析师兼可视化架构师。
基于第一阶段筛选出的高价值发现，请提供深度的解读，并规划可视化的具体路径。

【高价值发现列表】
{high_value_findings_str}

【你的任务】
1. 将发现转化为严谨的学术结论 (conclusion)。
2. 为每个发现规划一个直观的图表 (chart_type 支持: heatmap, bar, scatter, boxplot)。
3. 必须输出合法的 JSON 对象，不包含 Markdown 包裹符。

【输出格式】
{{
  "detailed_insights": [
    {{
      "metric": "指标名称",
      "conclusion": "深入解读 (约100字)",
      "anomaly": "是否涉及异常 (是/否，并简述)"
    }}
  ],
  "visualization_plan": [
    {{
      "chart_type": "图表类型",
      "title": "标题",
      "x_axis": "X轴含义",
      "y_axis": "Y轴含义",
      "plot_instruction": "给 Plotting Agent 的具体绘图建议"
    }}
  ]
}}"""

        response_2 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": detailed_prompt}],
            temperature=0.7
        )
        
        final_insights_str = response_2.choices[0].message.content.strip()
        # 处理可能的 Markdown 包裹
        if final_insights_str.startswith("```json"):
            final_insights_str = final_insights_str.split("```json")[1].split("```")[0].strip()
        
        final_insights = json.loads(final_insights_str)
        print("✅ 深度解读与绘图规划完成。")

        # ---------------------------------------------------------
        # 步骤 3: 存档与状态更新
        # ---------------------------------------------------------
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        insights_path = os.path.join(output_dir, "analysis_insights.json")
        with open(insights_path, 'w', encoding='utf-8') as f:
            json.dump(final_insights, f, ensure_ascii=False, indent=2)

        print(f"✅ 最终分析报告已保存至: {insights_path}")

        return {
            "analysis_insights": final_insights,
            "basic_stats": basic_stats,
            "semantic_stats": semantic_stats,
            "current_step": "analysis_agent"
        }

    except json.JSONDecodeError as e:
        error_msg = f"[Analysis Agent] JSON 解析失败: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error_logs": [error_msg], "current_step": "error"}
    except Exception as e:
        error_msg = f"[Analysis Agent] 运行失败: {str(e)}"
        import traceback
        traceback.print_exc()
        print(f"❌ {error_msg}")
        return {"error_logs": [error_msg], "current_step": "error"}