# analysis/comparison_analyzer.py
"""Сравнительный анализ вариантов стратегии"""
import pandas as pd


class ComparisonAnalyzer:
    """Сравнивает разные варианты параметров"""
    
    @staticmethod
    def compare_strategies(results: list) -> pd.DataFrame:
        """Сравнивает результаты стратегий"""
        df = pd.DataFrame(results)
        df = df.sort_values('total_profit', ascending=False)
        return df
    
    @staticmethod
    def rank_parameters(results_df: pd.DataFrame) -> dict:
        """Ранжирует параметры по влиянию на результат"""
        correlations = results_df.corr()['total_profit'].abs().sort_values(ascending=False)
        return correlations.to_dict()
