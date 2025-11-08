# trade_system/strategies/strategy_factory.py
"""Фабрика для создания экземпляров стратегий"""
import logging
from typing import Optional

from configuration.settings import StrategySettings
from trade_system.strategies.base_strategy import IStrategy
from trade_system.strategies.example_strategy import ExampleStrategy
from trade_system.strategies.seykota_kelly_strategy import SeykoaKellyStrategy

__all__ = ("StrategyFactory",)

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Фабрика стратегий - создает экземпляры по имени"""
    
    @staticmethod
    def new_factory(strategy_settings: StrategySettings) -> Optional[IStrategy]:
        """
        Создает новый экземпляр стратегии по настройкам
        
        Args:
            strategy_settings: Настройки стратегии
            
        Returns:
            Экземпляр стратегии или None если стратегия не найдена
        """
        strategy_name = strategy_settings.name
        
        logger.info(f"Создание стратегии: {strategy_name}")
        
        if strategy_name == "ExampleStrategy":
            return ExampleStrategy(strategy_settings)
        
        elif strategy_name == "SeykoaKellyStrategy":
            return SeykoaKellyStrategy(strategy_settings)
        
        else:
            logger.error(f"Неизвестная стратегия: {strategy_name}")
            return None
