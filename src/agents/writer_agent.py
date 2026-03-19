from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.tools.rag_retriever import retrieve_documents

def writer_agent_node(state):
    llm = ChatOpenAI(model="gpt-4o")
    
    retrieved_docs = retrieve_documents(state["topic"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的学术写作专家。结合分析结论和相关文献，撰写学术论文或报告。"),
        ("user", "主题：{topic}\n分析结论：{analysis_insights}\n相关文献：{retrieved_docs}\n请撰写 Markdown 格式的论文。")
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "topic": state["topic"],
        "analysis_insights": state["analysis_insights"],
        "retrieved_docs": retrieved_docs
    })
    
    return {
        **state,
        "thesis_draft": result.content,
        "current_step": "writer_agent"
    }
