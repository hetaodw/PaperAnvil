from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.tools.python_repl import execute_python_code

def plotting_agent_node(state):
    llm = ChatOpenAI(model="gpt-4o")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个数据可视化专家。根据分析结论绘制图表。"),
        ("user", "分析结论：{analysis_insights}\n请编写 Python 代码生成图表，并保存到 data/output/ 目录。")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"analysis_insights": state["analysis_insights"]})
    
    python_code = result.content
    execution_result = execute_python_code(python_code)
    
    return {
        **state,
        "plot_image_paths": execution_result.get("image_paths", []),
        "current_step": "plotting_agent"
    }
