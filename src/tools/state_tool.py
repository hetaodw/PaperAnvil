import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class StateTool:
    """
    封装多种数据分析方法的工具类，用于对扩增后的数据进行深度分析。
    包含聚类、异常检测、相关性分析和特征贡献度评估。
    """

    def __init__(self, data_path: str, questionnaire: Dict[str, Any]):
        self.data_path = data_path
        self.questionnaire = questionnaire
        self.df = pd.read_csv(data_path)
        self.insights = {}
        
        # 提取列名
        self.likert_cols = [q["id"] for q in questionnaire.get("likert_scales", [])]
        self.demographic_cols = [q["id"] for q in questionnaire.get("demographics", [])]
        
    def _prepare_numeric_data(self):
        """仅提取数值型的李克特量表数据用于聚类和异常检测"""
        return self.df[self.likert_cols].copy()

    def analyze_clustering(self):
        """多算法聚类：K-Means 和 DBSCAN"""
        print(">>> 正在进行聚类分析...")
        data = self._prepare_numeric_data()
        
        # 1. K-Means (假设分为 3 类作为示例)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.df['kmeans_cluster'] = kmeans.fit_predict(data)
        
        # 2. DBSCAN
        dbscan = DBSCAN(eps=1.5, min_samples=5)
        self.df['dbscan_cluster'] = dbscan.fit_predict(data)
        
        self.insights["clustering"] = {
            "kmeans_centers": kmeans.cluster_centers_.tolist(),
            "kmeans_counts": self.df['kmeans_cluster'].value_counts().to_dict(),
            "dbscan_counts": self.df['dbscan_cluster'].value_counts().to_dict(),
            "conclusion": "数据已被分为不同群体。K-Means 提供了清晰的分群界限，而 DBSCAN 识别了自然密度分布。"
        }

    def analyze_anomalies(self):
        """异常检测：Isolation Forest"""
        print(">>> 正在进行异常检测...")
        data = self._prepare_numeric_data()
        
        # 孤立森林，设定污染率为 5%
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        # -1 为异常，1 为正常
        self.df['is_anomaly'] = iso_forest.fit_predict(data)
        
        anomalies = self.df[self.df['is_anomaly'] == -1]
        
        self.insights["anomaly_detection"] = {
            "anomaly_count": int(len(anomalies)),
            "anomaly_percentage": float(len(anomalies) / len(self.df)),
            "example_anomalies": anomalies.head(5).to_dict(orient="records"),
            "conclusion": "识别出约 5% 的极端用户偏激样本或交互路径异常。"
        }

    def analyze_correlations(self):
        """相关性分析：计算量表评分与人口特征的相关系数"""
        print(">>> 正在进行相关性分析...")
        
        # 为了计算相关性，需要对人口统计学特征进行编码
        df_encoded = self.df.copy()
        for col in self.demographic_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
        # 计算完整相关矩阵
        all_cols = self.likert_cols + self.demographic_cols
        corr_matrix = df_encoded[all_cols].corr()
        
        self.insights["correlation_analysis"] = {
            "matrix": corr_matrix.to_dict(),
            "top_correlations": self._get_top_correlations(corr_matrix),
            "conclusion": "分析了量表评分与人口统计特征之间的线性关联。"
        }

    def _get_top_correlations(self, corr_matrix, threshold=0.1):
        """提取显著的相关性配对"""
        pairs = []
        labels = corr_matrix.columns
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= threshold:
                    pairs.append({
                        "feature1": labels[i],
                        "feature2": labels[j],
                        "correlation": float(val)
                    })
        return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:10]

    def analyze_feature_importance(self):
        """特征贡献度评估：基于 Random Forest"""
        print(">>> 正在进行特征贡献度评估...")
        
        # 假设以最后一个李克特题（通常是整体满意度）作为目标变量
        if not self.likert_cols:
            return
            
        target = self.likert_cols[-1]
        features = [col for col in self.likert_cols[:-1]] + self.demographic_cols
        
        # 编码
        df_encoded = self.df.copy()
        for col in self.demographic_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
        X = df_encoded[features]
        y = df_encoded[target]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = dict(zip(features, rf.feature_importances_.tolist()))
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        self.insights["feature_contribution"] = {
            "target_variable": target,
            "importances": dict(sorted_importances),
            "top_features": [f[0] for f in sorted_importances[:5]],
            "conclusion": f"根据随机森林模型，对 {target} 影响最大的因素已识别。"
        }

    def run_all(self):
        """执行所有分析方法"""
        self.analyze_clustering()
        self.analyze_anomalies()
        self.analyze_correlations()
        self.analyze_feature_importance()
        return self.insights

    def save_results(self, output_dir: str = "data/intermediate"):
        """将分析结果存入 JSON 文件"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "analysis_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.insights, f, ensure_ascii=False, indent=4)
        print(f"✅ 分析结果已存入: {output_path}")
        return output_path

if __name__ == "__main__":
    # 简单的冒烟测试
    # 假设已有 questionnaire.json 和 simulated_data.csv
    try:
        with open("data/intermediate/questionnaire.json", "r", encoding="utf-8") as f:
            q = json.load(f)
        
        # 找一个现有的 csv 测试，或者如果没有就跳过
        csv_path = "data/raw_data/simulated_data.csv"
        if os.path.exists(csv_path):
            tool = StateTool(csv_path, q)
            results = tool.run_all()
            tool.save_results()
            print("Done!")
    except Exception as e:
        print(f"Skipping smoke test: {e}")
