"""Конфигурация системы логирования для анализа сделок в бэктесте.

Отслеживает:
- Входы в позиции (LONG/SHORT)
- Параметры стопов (ATR, Percent, Swing)
- Закрытия позиций (Profit/Loss, причины)
- Фильтры и пропуски сигналов
"""

import logging
import logging.handlers
from pathlib import Path
import sys
from datetime import datetime

# Создание директории для логов
LOG_DIR = Path('logs_backtest')
LOG_DIR.mkdir(exist_ok=True)

# Уровни логирования для разных компонентов
COMPONENT_LEVELS = {
    'signal_detection': logging.DEBUG,      # Все сигналы и фильтры
    'entry_analysis': logging.DEBUG,         # Детали входов
    'stop_loss_calc': logging.DEBUG,        # Расчёты стопов
    'exit_analysis': logging.DEBUG,          # Закрытия и прибыль
    'kelly_percent': logging.DEBUG,          # Расчёт Kelly
    'trade_summary': logging.INFO,           # Итоги сделок
}

def setup_logger(name, component=None):
    """Настройка логгера с разными уровнями для компонентов."""
    logger = logging.getLogger(name)
    
    level = COMPONENT_LEVELS.get(component, logging.INFO)
    logger.setLevel(level)
    
    # Удалить существующие handlers
    logger.handlers.clear()
    
    # Формат для файла - максимально подробный
    file_format = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Формат для консоли - кратче
    console_format = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Файловый handler - все логи
    file_handler = logging.FileHandler(
        LOG_DIR / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Консольный handler - выше INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger

# Создание логгеров для разных компонентов анализа
logger_signal = setup_logger('signal_detection', 'signal_detection')
logger_entry = setup_logger('entry_analysis', 'entry_analysis')
logger_stop = setup_logger('stop_loss_calc', 'stop_loss_calc')
logger_exit = setup_logger('exit_analysis', 'exit_analysis')
logger_kelly = setup_logger('kelly_percent', 'kelly_percent')
logger_trade = setup_logger('trade_summary', 'trade_summary')

# Основной логгер для общей информации
logger_main = setup_logger('backtest')
