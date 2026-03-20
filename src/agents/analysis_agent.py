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
        # 从 State 获取数据，如果没有，再找本地固定路径防抖动
        responses_path = "data/intermediate/open_ended_responses.json"
        
        if not raw_data_path or not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"找不到原始数据文件: {raw_data_path}")
        if not questionnaire:
            raise ValueError("State 中缺少问卷 (questionnaire) 信息")

        # 【重点修复】从内存状态获取开放题答案，并强制同步回落盘供 Tool 读取
        open_ended_responses = state.get("open_ended_detailed_responses", [])
        if open_ended_responses:
            os.makedirs(os.path.dirname(responses_path), exist_ok=True)
            with open(responses_path, "w", encoding="utf-8") as f:
                json.dump(open_ended_responses, f, ensure_ascii=False, indent=2)
            print(">>> ✅ 已成功同步最新开放题数据到本地供语义分析使用。")


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
        
        prompt_template_1 = state.get("prompts", {}).get("analysis_filter_prompt", "")
        if not prompt_template_1: prompt_template_1 = "分析：{raw_stats_context}"
        filter_prompt = prompt_template_1.format(
            raw_stats_context=json.dumps(raw_stats_context, ensure_ascii=False)[:10000]
        )

        response_1 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": filter_prompt}],
            temperature=0.3
        )
        
        msg_1 = response_1.choices[0].message
        if hasattr(msg_1, 'reasoning_content') and msg_1.reasoning_content:
            print(f"\n[Analysis Agent - 洞察筛选 思维链]\n{msg_1.reasoning_content}\n")
        
        high_value_findings_str = msg_1.content.strip()
        print(f"\n[Analysis Agent - 洞察筛选 LLM 输出]\n{high_value_findings_str}\n")
        print("✅ 洞察筛选完成。")

        # ---------------------------------------------------------
        # 步骤 2: 详细解读与图表规划
        # ---------------------------------------------------------
        print(">>> 步骤 2: 正在进行深度报告撰写规划与绘图指令生成...")
        
        prompt_template_2 = state.get("prompts", {}).get("analysis_detailed_prompt", "")
        if not prompt_template_2: prompt_template_2 = "基于发现 {high_value_findings_str} 进行解读。"
        detailed_prompt = prompt_template_2.format(
            high_value_findings_str=high_value_findings_str
        )

        response_2 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": detailed_prompt}],
            temperature=0.7
        )
        
        msg_2 = response_2.choices[0].message
        if hasattr(msg_2, 'reasoning_content') and msg_2.reasoning_content:
            print(f"\n[Analysis Agent - 解读与规划 思维链]\n{msg_2.reasoning_content}\n")
            
        final_insights_str = msg_2.content.strip()
        print(f"\n[Analysis Agent - 解读与规划 LLM 输出]\n{final_insights_str}\n")
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