# ml_analysis/feature_extractor.py
"""Извлечение признаков для ML"""
import numpy as np


class FeatureExtractor:
    """Извлекает признаки из свечей для ML"""
    
    @staticmethod
    def extract_candlestick_features(candles: list) -> dict:
        """Извлекает признаки свечей"""
        last = candles[-1]
        
        body = abs(last['close'] - last['open'])
        total_range = last['high'] - last['low']
        upper_shadow = last['high'] - max(last['open'], last['close'])
        lower_shadow = min(last['open'], last['close']) - last['low']
        
        return {
            'body_ratio': body / total_range if total_range > 0 else 0,
            'upper_shadow_ratio': upper_shadow / body if body > 0 else 0,
            'lower_shadow_ratio': lower_shadow / body if body > 0 else 0,
            'is_bullish': 1 if last['close'] > last['open'] else 0,
            'volume_ratio': last['volume'] / np.mean([c['volume'] for c in candles[-20:]]) if len(candles) >= 20 else 1
        }
    
    @staticmethod
    def extract_ema_features(candles: list, short_period: int = 18, long_period: int = 50) -> dict:
        """Извлекает EMA признаки"""
        closes = [c['close'] for c in candles]
        
        ema_short = np.mean(closes[-short_period:]) if len(closes) >= short_period else closes[-1]
        ema_long = np.mean(closes[-long_period:]) if len(closes) >= long_period else closes[-1]
        
        return {
            'ema_distance': (ema_short - ema_long) / ema_long if ema_long > 0 else 0,
            'price_above_ema_short': 1 if closes[-1] > ema_short else 0,
            'price_above_ema_long': 1 if closes[-1] > ema_long else 0
        }
