import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List

class BasicStatsTool:
    """
    基础统计工具类，用于计算频率分布、均值、方差等描述性统计指标。
    """

    def __init__(self, data_path: str, questionnaire: Dict[str, Any]):
        self.data_path = data_path
        self.questionnaire = questionnaire
        self.df = pd.read_csv(data_path)
        self.results = {}
        
        # 提取列名
        self.likert_cols = [q["id"] for q in questionnaire.get("likert_scales", [])]
        self.demographic_cols = [q["id"] for q in questionnaire.get("demographics", [])]

    def analyze_demographics(self):
        """计算人口统计学特征的频率分布"""
        print(">>> 正在统计人口学特征分布...")
        demographics_stats = {}
        for col in self.demographic_cols:
            if col in self.df.columns:
                counts = self.df[col].value_counts()
                percentages = self.df[col].value_counts(normalize=True) * 100
                
                demographics_stats[col] = {
                    "counts": counts.to_dict(),
                    "percentages": percentages.to_dict()
                }
        self.results["demographics_distribution"] = demographics_stats

    def analyze_likert_stats(self):
        """计算李克特量表的描述性统计和分布"""
        print(">>> 正在统计量表评分分布...")
        likert_stats = {}
        for col in self.likert_cols:
            if col in self.df.columns:
                # 基础统计量
                desc = self.df[col].describe()
                # 1-5 分布 (确保所有 1-5 都有记录，即使计数为 0)
                dist = self.df[col].value_counts().reindex(range(1, 6), fill_value=0)
                # 转换索引为字符串以保持 JSON 兼容性
                dist.index = dist.index.astype(str)
                
                likert_stats[col] = {
                    "mean": float(desc["mean"]),
                    "std": float(desc["std"]),
                    "median": float(self.df[col].median()),
                    "min": float(desc["min"]),
                    "max": float(desc["max"]),
                    "distribution": dist.to_dict()
                }
        self.results["likert_stats"] = likert_stats

    def run_all(self):
        """执行所有统计分析"""
        self.analyze_demographics()
        self.analyze_likert_stats()
        return self.results

    def save_results(self, output_dir: str = "data/intermediate"):
        """将统计结果存入 JSON 文件"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "basic_stats.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        print(f"✅ 基础统计结果已存入: {output_path}")
        return output_path

if __name__ == "__main__":
    # 冒烟测试
    try:
        with open("data/intermediate/questionnaire.json", "r", encoding="utf-8") as f:
            q = json.load(f)
        csv_path = "data/raw_data/simulated_data.csv"
        if os.path.exists(csv_path):
            tool = BasicStatsTool(csv_path, q)
            tool.run_all()
            tool.save_results()
    except Exception as e:
        print(f"Skipping smoke test: {e}")
