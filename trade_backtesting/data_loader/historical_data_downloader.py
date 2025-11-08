# data_loader/historical_data_downloader.py
"""Загрузка исторических данных из T-Invest API"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tinkoff.invest import Client, CandleInterval
import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalDataDownloader:
    """Загрузка и сохранение исторических данных"""
    
    def __init__(self, token: str, app_name: str, cache_dir: str = "historical_data"):
        self.token = token
        self.app_name = app_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_candles(self, figi: str, interval: CandleInterval, days: int = 1095):
        """Загружает свечи за указанный период (по умолчанию 3 года)"""
        end = datetime.now()
        start = end - timedelta(days=days)
        
        logger.info(f"Загрузка {days} дней данных для {figi}...")
        
        all_candles = []
        
        with Client(token=self.token, app_name=self.app_name) as client:
            chunk_days = 30
            current_start = start
            
            while current_start < end:
                current_end = min(current_start + timedelta(days=chunk_days), end)
                
                candles = client.market_data.get_candles(
                    figi=figi,
                    from_=current_start,
                    to=current_end,
                    interval=interval
                )
                
                all_candles.extend(candles.candles)
                logger.info(f"Загружено {len(candles.candles)} свечей ({current_start.date()} - {current_end.date()})")
                
                current_start = current_end
        
        logger.info(f"✅ Всего загружено: {len(all_candles)} свечей")
        return all_candles
    
    def save_to_csv(self, candles, filename: str):
        """Сохраняет свечи в CSV"""
        data = []
        for candle in candles:
            data.append({
                'time': candle.time,
                'open': float(candle.open.units + candle.open.nano / 1e9),
                'high': float(candle.high.units + candle.high.nano / 1e9),
                'low': float(candle.low.units + candle.low.nano / 1e9),
                'close': float(candle.close.units + candle.close.nano / 1e9),
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        filepath = self.cache_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Сохранено в {filepath}")
        return filepath
