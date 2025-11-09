"""Функции диагностики торговых сделок.

Отправляют подробные логи входов, выходов, стопов и расчётов Kelly.
"""

from logging_config import logger_entry, logger_stop, logger_exit, logger_signal


def log_entry_analysis(trade_num, ticker, signal_type, entry_price, position_size, kelly_percent):
    """Логирует детали входа в сделку."""
    logger_entry.debug(f"""
    Трейд #{trade_num} | {ticker} | {signal_type}
    Цена входа: {entry_price:.2f}
    Позиция: {position_size} лотов ({kelly_percent:.1f}% Kelly)
    """)


def log_stop_loss_calculation(trade_num, ticker, atr_value, atr_multiplier, atr_stop, 
                              percent_stop, swing_stop, final_stop):
    """Логирует расчет стоп-лосса."""
    logger_stop.debug(f"""
    Настройка стопов трейд #{trade_num} | {ticker}
    ATR: {atr_value:.2f} х {atr_multiplier} = {atr_stop:.2f}
    Percent: {percent_stop:.2f}
    Swing: {swing_stop:.2f}
    ВЫБРАННЫЙ STOP: {final_stop:.2f}
    """)


def log_exit_analysis(trade_num, ticker, exit_reason, entry_price, exit_price, profit_loss, 
                       commission=0, stop_distance=0):
    """Логирует детали выхода из сделки."""
    logger_exit.debug(f"""
    Прибыль/Убыток трейд #{trade_num} | {ticker} | {exit_reason}
    Выход: {entry_price:.2f} -> {exit_price:.2f}
    P&L: {profit_loss:.2f} руб. (comm: {commission:.2f})
    Расстояние до стопа: {stop_distance:.2f}
    """)


def log_signal_filter_rejection(ticker, candle_idx, reason, details=None):
    """Логирует причины фильтрации сигналов."""
    msg = f"[{ticker} @{candle_idx}] SIGNAL REJECTED: {reason}"
    if details:
        msg += f" | {details}"
    logger_signal.debug(msg)


def log_kelly_calculation(trade_num, ticker, win_rate, profit_factor, kelly_percent, position_size):
    """Логирует расчёт Kelly%."""
    logger_signal.debug(f"""
    Kelly% расчёт трейд #{trade_num} | {ticker}
    Win Rate: {win_rate:.1%} | Profit Factor: {profit_factor:.2f}
    Kelly%: {kelly_percent:.1f}% -> Позиция: {position_size} лотов
    """)
