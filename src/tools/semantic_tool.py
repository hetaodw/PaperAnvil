import json
import os
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import dashscope
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SemanticTool:
    """
    语义分析工具类，提供主题建模 (LDA)、细粒度情感分析 (ABSA) 和语义向量聚类功能。
    """

    def __init__(self, responses_path: str):
        self.responses_path = responses_path
        with open(responses_path, "r", encoding="utf-8") as f:
            self.responses_data = json.load(f)
        
        # 提取所有文本内容
        self.texts = []
        for entry in self.responses_data:
            for q_id, text in entry.get("responses", {}).items():
                if q_id.startswith("o"): # 只处理开放题
                    self.texts.append(text)
        
        self.results = {}
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )

    def analyze_topic_modeling(self, n_topics: int = 5):
        """语义主题建模 (LDA)"""
        print(f">>> 正在进行主题建模 (目标主题数: {n_topics})...")
        if not self.texts:
            return
            
        vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words=None) # 中文通常需要分词，此处简化处理或假设已分词
        tf = vectorizer.fit_transform(self.texts)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tf)
        
        # 提取关键词
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10 - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                "topic_id": topic_idx,
                "keywords": top_words
            })
            
        # 调用 LLM 对主题进行人性化总结
        summary_prompt = f"以下是从调研文本中提取的若干主题关键词，请为每个主题总结一个核心名称（如‘网络波动下的登录失败’）和一段简短描述：\n{json.dumps(topics, ensure_ascii=False)}"
        
        try:
            completion = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
                messages=[{"role": "user", "content": summary_prompt}]
            )
            self.results["topic_modeling"] = {
                "raw_topics": topics,
                "interpretation": completion.choices[0].message.content.strip()
            }
        except Exception as e:
            print(f"主题总结失败: {e}")
            self.results["topic_modeling"] = {"raw_topics": topics}

    def analyze_absa(self):
        """细粒度情感分析 (ABSA)"""
        print(">>> 正在进行细粒度情感分析 (ABSA)...")
        if not self.texts:
            return

        # 抽样处理（如果文本过多，LLM 调用成本高）
        sample_texts = self.texts[:20] 
        absa_results = []
        
        prompt = """分析以下调研文本中的“实体（Aspect）”和对应的“情感倾向（Sentiment）”。
输出格式要求为 JSON 数组：[{"aspect": "...", "sentiment": "正面/负面/中性", "reason": "..."}]。
文本列表：
""" + "\n".join([f"- {t}" for t in sample_texts])

        try:
            completion = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
                messages=[{"role": "user", "content": prompt}]
            )
            content = completion.choices[0].message.content.strip()
            clean_json = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
            absa_results = json.loads(clean_json)
        except Exception as e:
            print(f"ABSA 分析失败: {e}")

        self.results["absa"] = absa_results

    def analyze_semantic_clustering(self):
        """语义向量聚类 (Embedding + K-Means)"""
        print(">>> 正在进行语义向量聚类...")
        if not self.texts:
            return

        # 1. 获取 Embeddings
        embeddings = []
        try:
            # 使用阿里云百炼 text-embedding-v4
            # 注意：dashscope 会自动从环境或配置中读取 api_key，此处不显式硬编码
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            
            # 为了效率，可以尝试 batch 处理（dashscope 支持 list input）
            batch_size = 25
            for i in range(0, min(len(self.texts), 100), batch_size): # 限制前 100 条
                batch_input = self.texts[i : i + batch_size]
                resp = dashscope.TextEmbedding.call(
                    model="text-embedding-v4",
                    input=batch_input
                )
                if resp.status_code == 200:
                    # 提取 embedding 向量
                    batch_embeddings = [record['embedding'] for record in resp.output['embeddings']]
                    # 确保不超出当前 batch 的长度（预防 Mock 或 API 返回超量）
                    embeddings.extend(batch_embeddings[:len(batch_input)])
                else:
                    print(f"DashScope Embedding 失败: {resp.code} - {resp.message}")
                    break
        except Exception as e:
            print(f"获取 Embedding 异常: {e}")
            return

        # 2. K-Means 聚类
        if embeddings:
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            cluster_summaries = []
            for i in range(n_clusters):
                cluster_texts = [self.texts[j] for j in range(len(clusters)) if clusters[j] == i]
                cluster_summaries.append({
                    "cluster_id": i,
                    "sample_count": len(cluster_texts),
                    "representative_text": cluster_texts[0] if cluster_texts else ""
                })
            
            self.results["semantic_clustering"] = cluster_summaries

    def run_all(self):
        """执行所有分析"""
        self.analyze_topic_modeling()
        self.analyze_absa()
        self.analyze_semantic_clustering()
        return self.results

    def save_results(self, output_dir: str = "data/intermediate"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "semantic_analysis.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        print(f"✅ 语义分析结果已存入: {output_path}")
        return output_path

if __name__ == "__main__":
    # 暂不执行测试逻辑，等待用户修改指令
    pass
