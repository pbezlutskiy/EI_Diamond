# data_loader/data_cache_manager.py
"""Управление кешем исторических данных"""
import pandas as pd
from pathlib import Path


class DataCacheManager:
    """Загрузка данных из кеша CSV"""
    
    def __init__(self, cache_dir: str = "historical_data"):
        self.cache_dir = Path(cache_dir)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Загружает CSV в DataFrame"""
        filepath = self.cache_dir / filename
        df = pd.read_csv(filepath, parse_dates=['time'])
        return df
    
    def get_available_files(self) -> list:
        """Возвращает список доступных CSV файлов"""
        return list(self.cache_dir.glob("*.csv"))
