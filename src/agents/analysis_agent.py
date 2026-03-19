from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.tools.python_repl import execute_python_code

def analysis_agent_node(state):
    llm = ChatOpenAI(model="gpt-4o")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个数据科学专家。执行机器学习聚类、异常检测等分析任务。"),
        ("user", "数据路径：{raw_data_path}\n请编写 Python 代码进行数据分析，并将结果保存到 data/processed/ 目录。")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"raw_data_path": state["raw_data_path"]})
    
    python_code = result.content
    execution_result = execute_python_code(python_code)
    
    return {
        **state,
        "analysis_insights": execution_result,
        "current_step": "analysis_agent"
    }
