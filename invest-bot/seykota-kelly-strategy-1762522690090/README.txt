
Seykota-Kelly Strategy - Полная система бэктестинга
====================================================

СТРУКТУРА АРХИВА:
- invest-bot/ - файлы для EIDiamond/invest-bot
- trade_backtesting/ - файлы для EIDiamond/trade_backtesting

УСТАНОВКА:

1. EIDiamond/invest-bot:
   cd путь/к/invest-bot
   Скопируйте файлы из архива/invest-bot/ в соответствующие папки

2. EIDiamond/trade_backtesting:
   cd путь/к/trade_backtesting
   Скопируйте файлы из архива/trade_backtesting/ в соответствующие папки

3. Git коммит:
   git add .
   git commit -m "Add Seykota-Kelly strategy with backtesting system"
   git push origin main

4. Запуск:
   cd trade_backtesting
   python main_backtest_runner.py

ЗАВИСИМОСТИ:
pip install pandas numpy scikit-learn plotly tinkoff-invest

Создано: 07.11.2025, 17:38:10
