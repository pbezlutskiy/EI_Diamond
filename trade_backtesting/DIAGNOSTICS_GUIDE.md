# Логирование анализа сделок (Entry/Exit Diagnostics)

## Объектив: Понимание почему мало сделок и неадекватные стопы

## Файлы диагностики

### 1. `logging_config.py` - Конфигурация логирования

Настраивает разные логгеры для каждого компонента анализа:

- **logger_signal**: DEBUG - все сигналы и фильтры
- **logger_entry**: DEBUG - детали входов
- **logger_stop**: DEBUG - расчёты стопов
- **logger_exit**: DEBUG - закрытия и прибыль
- **logger_kelly**: DEBUG - расчёт Kelly%
- **logger_trade**: INFO - итоги сделок

Логи пишутся:
- В файлы `logs_backtest/` (все данные делали)
- На консоль (только INFO и выше)

### 2. `trade_diagnostics.py` - Функции диагностики

Функции для логирования:

```python
log_entry_analysis(trade_num, ticker, signal_type, entry_price, position_size, kelly_percent)
```
Логирует детали входа:
- Номер трейда и тикер
- Тип сигнала (LONG/SHORT)
- Цена входа
- Размер позиции и Kelly%

```python
log_stop_loss_calculation(trade_num, ticker, atr_value, atr_multiplier, atr_stop, percent_stop, swing_stop, final_stop)
```
Рассширяет расчёт стоп:
- ATR и его множитель
- Процентный стоп
- Swing стоп
- Выбранный стоп

```python
log_exit_analysis(trade_num, ticker, exit_reason, entry_price, exit_price, profit_loss, commission, stop_distance)
```
Логирует выход из сделки

## Как модифицировать main_backtest_runner.py

1. Адд импорты вку конца:
```python
from trade_diagnostics import (log_entry_analysis, log_stop_loss_calculation, 
                                log_exit_analysis, log_kelly_calculation)
```

2. В метод `_simulate_trading` добавить вызовы диагностики около выходов сигналов и расчётов стопов

## Локация логов

Все логи находятся в `logs_backtest/` директории с временным штампом в имени файла.
