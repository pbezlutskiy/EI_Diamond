# analysis/correlation_analyzer.py
"""Корреляционный анализ параметров"""
import pandas as pd


class CorrelationAnalyzer:
    """Анализ корреляций параметров стратегии с результатами"""
    
    @staticmethod
    def calculate_correlation_matrix(results_df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитывает корреляционную матрицу"""
        corr_matrix = results_df.corr()
        return corr_matrix
    
    @staticmethod
    def find_best_parameters(results_df: pd.DataFrame, metric: str = 'total_profit') -> dict:
        """Находит лучшие параметры"""
        best_row = results_df.loc[results_df[metric].idxmax()]
        return best_row.to_dict()
