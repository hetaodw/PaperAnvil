import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

class RAGRetriever:
    """
    从向量数据库检索相关文献的工具类。
    使用 ChromaDB 作为向量存储后端。
    """
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "knowledge_base/vector_store"):
        """
        初始化 RAG 检索器。
        
        Args:
            collection_name: ChromaDB 集合名称
            persist_directory: 向量数据库持久化目录
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def retrieve_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        根据查询检索相关文档。
        
        Args:
            query: 查询文本
            n_results: 返回的结果数量，默认为 5
            
        Returns:
            检索到的文档列表，每个文档包含 {
                "content": str,      # 文档内容
                "metadata": dict,    # 文档元数据
                "distance": float    # 相似度距离
            }
            
        Raises:
            Exception: 检索过程中抛出的异常
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            documents = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            return documents
            
        except Exception as e:
            print(f"检索文档时出错: {str(e)}")
            return []
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """
        添加文档到向量数据库。
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            
        Raises:
            Exception: 添加文档过程中抛出的异常
        """
        try:
            doc_id = f"doc_{len(self.collection.get()['ids'])}"
            self.collection.add(
                documents=[content],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"添加文档时出错: {str(e)}")

_retriever = None

def retrieve_documents(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    从向量数据库检索相关文档的便捷函数。
    
    Args:
        query: 查询文本
        n_results: 返回的结果数量，默认为 5
        
    Returns:
        检索到的文档列表
    """
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever.retrieve_documents(query, n_results)
