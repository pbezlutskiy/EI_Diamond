# main_backtest_runner.py
"""Главный запуск РЕАЛЬНОГО бэктеста с интеграцией всех компонентов"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append('../invest-bot')

from tinkoff.invest import Client, CandleInterval, HistoricCandle, Quotation
from trade_system.strategies.seykota_kelly_strategy import SeykoaKellyStrategy
from configuration.settings import StrategySettings
from trade_system.signal import SignalType
from detailed_backtest_visualizer import DetailedBacktestVisualizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealBacktestRunner:
    """РЕАЛЬНЫЙ бэктест на исторических данных"""
    
    def __init__(self, token: str, app_name: str):
        self.token = token
        self.app_name = app_name
    
    def run(self, figi: str, ticker: str, strategy_settings: dict):
        """Запускает полный цикл бэктеста"""
        logger.info("=" * 70)
        logger.info(f"🚀 ЗАПУСК РЕАЛЬНОГО БЭКТЕСТА: {ticker}")
        logger.info("=" * 70)
        
        # 1. Загрузка данных
        candles = self._load_data(figi, ticker)
        logger.info(f"✅ Данные загружены: {len(candles)} свечей")
        
        # 2. Создание стратегии
        strategy = self._create_strategy(figi, ticker, strategy_settings)
        
        # 3. Симуляция торговли
        trades, equity, kelly_history = self._simulate_trading(strategy, candles)  # ← ИЗМЕНИТЬ
        logger.info(f"✅ Симуляция завершена: {len(trades)} сделок")
        
        # 4. Расчет метрик
        metrics = self._calculate_metrics(trades, equity)

        # 5. Генерация HTML отчета
        self._generate_html_report(ticker, metrics, trades, equity)

        # 6. Детальный отчет с графиками
        DetailedBacktestVisualizer.generate_detailed_report(
            ticker=ticker,
            candles_data=candles,
            trades=trades,
            equity=equity,
            metrics=metrics,
            kelly_history=kelly_history,  # ← ДОБАВИТЬ
            output_file=f"backtest_results/{ticker}_DETAILED.html"
        )

        logger.info(f"📊 Детальный отчет: backtest_results/{ticker}_DETAILED.html")

        # Вывод результатов
        logger.info("\n" + "=" * 70)

        logger.info("📊 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
        logger.info(f"   💰 Total Profit: {metrics['total_profit']:.2f} ₽")
        logger.info(f"   📈 Win Rate: {metrics['win_rate']*100:.1f}%")
        logger.info(f"   🎯 Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"   📉 Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
        logger.info(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"   🔢 Total Trades: {metrics['total_trades']}")
        logger.info("=" * 70)
        
        return metrics
    
    def _load_data(self, figi: str, ticker: str):
        """Загружает или скачивает исторические данные"""
        import pandas as pd
        
        filepath = Path(f"historical_data/{ticker}_3years.csv")
        
        # Если файл существует - загружаем из кеша
        if filepath.exists():
            logger.info(f"📂 Загрузка из кеша: {filepath}")
            df = pd.read_csv(filepath, parse_dates=['time'])
            return df.to_dict('records')
        
        # Иначе загружаем из API
        logger.info("📥 Загрузка данных из T-Invest API (3 года, часовые свечи)...")
        Path("historical_data").mkdir(exist_ok=True)
        
        end = datetime.now()
        start = end - timedelta(days=30)  # 3 года
        all_candles = []
        
        with Client(token=self.token, app_name=self.app_name) as client:
            current = start
            
            while current < end:
                next_date = min(current + timedelta(days=30), end)
                
                logger.info(f"   Загрузка: {current.date()} - {next_date.date()}")
                
                candles = client.market_data.get_candles(
                    figi=figi,
                    from_=current,
                    to=next_date,
                    interval=CandleInterval.CANDLE_INTERVAL_HOUR
                )
                
                all_candles.extend(candles.candles)
                current = next_date
                
                logger.info(f"   Всего загружено: {len(all_candles)} свечей")
        
        # Сохраняем в CSV
        data = []
        for c in all_candles:
            data.append({
                'time': c.time,
                'open': float(c.open.units + c.open.nano / 1e9),
                'high': float(c.high.units + c.high.nano / 1e9),
                'low': float(c.low.units + c.low.nano / 1e9),
                'close': float(c.close.units + c.close.nano / 1e9),
                'volume': c.volume
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Данные сохранены: {filepath}")
        
        return data
    
    def _create_strategy(self, figi: str, ticker: str, strategy_settings: dict):
        """Создает экземпляр стратегии"""
        return SeykoaKellyStrategy(figi, ticker, strategy_settings)

    
    def _simulate_trading(self, strategy, candles_data: list):
    """Симулирует торговлю на исторических данных (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)"""
    logger.info("\n🎮 НАЧАЛО СИМУЛЯЦИИ ТОРГОВЛИ (FAST MODE)")
    
    import numpy as np
    import pandas as pd
    
    # ШАГ 1: Конвертируем ВСЕ данные ОДИН РАЗ
    all_candles_hist = self._convert_to_historic_candles(candles_data)
    
    # ШАГ 2: Предрасчитываем ATR для ВСЕХ свечей (для стоп-лосса)
    df = pd.DataFrame(candles_data)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    # Рассчитываем ATR заранее
    atr_values = self._calculate_atr_vectorized(highs, lows, closes, strategy.atr_period)
    
    trades = []
    equity = [10000]
    position = None
    kelly_history = []
    
    min_candles = strategy.min_candles
    
    for i in range(min_candles, len(candles_data)):
        current_candle = candles_data[i]
        
        # Проверка стоп-лосса
        if position:
            if self._check_stop_hit(current_candle, position):
                trade = self._close_position(position, current_candle, 'stop')
                trades.append(trade)
                equity.append(equity[-1] + trade['profit'])
                
                logger.info(f" 💥 Сделка #{len(trades)}: STOP, profit={trade['profit']:.2f} ₽")
                
                strategy.record_trade_result(
                    signal_type=position['signal_type'],
                    entry_price=position['entry_price'],
                    exit_price=trade['exit_price'],
                    entry_time=position['entry_time'],
                    exit_time=current_candle['time']
                )
                position = None
        
        # ✅ ИСПОЛЬЗУЕМ ОРИГИНАЛЬНЫЙ МЕТОД СТРАТЕГИИ!
        if position is None:
            # Передаём последние i+1 свечей в стратегию
            candles_subset = all_candles_hist[:i+1]
            signal = strategy.analyze_candles(candles_subset)  # ← ПРАВИЛЬНО!
            
            if signal:
                kelly_pct = strategy._kelly_calculator()
                kelly_history.append(kelly_pct)
                position_size = max(1, int(equity[-1] * kelly_pct / current_candle['close']))
                
                position = {
                    'entry_price': current_candle['close'],
                    'entry_time': current_candle['time'],
                    'signal_type': signal.signal_type,
                    'stop_loss': float(signal.stop_loss_level),
                    'position_size': position_size
                }
                logger.info(f" 💰 Kelly%={kelly_pct*100:.1f}%, Size={position_size} lots")
                logger.info(f" 📈 Открытие #{len(trades)+1}: {signal.signal_type.name} @ {current_candle['close']:.2f}")
    
    # Закрываем последнюю позицию
    if position:
        trade = self._close_position(position, candles_data[-1], 'end')
        trades.append(trade)
        equity.append(equity[-1] + trade['profit'])
        logger.info(f" ⏹️ Закрытие последней позиции: profit={trade['profit']:.2f} ₽")
    
    logger.info(f"✅ Обработано {len(candles_data)} свечей в FAST MODE")
    return trades, equity, kelly_history

    
    def _calculate_ema_vectorized(self, data: np.ndarray, period: int) -> np.ndarray:
        """Векторизованный расчет EMA"""
        import pandas as pd
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    def _calculate_atr_vectorized(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Векторизованный расчет ATR"""
        import pandas as pd
        
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        true_range[0] = high_low[0]  # Первое значение
        
        atr = pd.Series(true_range).rolling(window=period).mean().values
        return atr
    
    def _check_signal_fast(self, ema_short_curr, ema_long_curr, ema_short_prev, ema_long_prev, 
                           candle, atr, strategy):
        """Быстрая проверка сигналов без полного analyze_candles"""
        from trade_system.signal import SignalType
        
        # Проверка пересечения EMA
        long_cross = (ema_short_prev <= ema_long_prev) and (ema_short_curr > ema_long_curr)
        short_cross = (ema_short_prev >= ema_long_prev) and (ema_short_curr < ema_long_curr)
        
        if not (long_cross or short_cross):
            return None
        
        # Расчет стоп-лосса
        if long_cross:
            # Находим swing low (минимум за последние swing_period свечей)
            # Упрощенная версия - используем ATR
            stop_loss = candle['close'] - (atr * strategy.atr_multiplier)
            
            return {
                'type': SignalType.LONG,
                'stop_loss': float(stop_loss)
            }
        else:  # short_cross
            stop_loss = candle['close'] + (atr * strategy.atr_multiplier)
            
            return {
                'type': SignalType.SHORT,
                'stop_loss': float(stop_loss)
            }

    
    def _convert_to_historic_candles(self, candles_dict: list):
        """Конвертирует dict в HistoricCandle для стратегии"""
        result = []
        for c in candles_dict:
            hc = HistoricCandle(
                open=Quotation(units=int(c['open']), nano=int((c['open'] % 1) * 1e9)),
                high=Quotation(units=int(c['high']), nano=int((c['high'] % 1) * 1e9)),
                low=Quotation(units=int(c['low']), nano=int((c['low'] % 1) * 1e9)),
                close=Quotation(units=int(c['close']), nano=int((c['close'] % 1) * 1e9)),
                volume=c['volume'],
                time=c['time'],
                is_complete=True
            )
            result.append(hc)
        return result
    
    def _check_stop_hit(self, candle: dict, position: dict) -> bool:
        """Проверяет попадание в стоп-лосс"""
        if position['signal_type'] == SignalType.LONG:
            return candle['low'] <= position['stop_loss']
        else:
            return candle['high'] >= position['stop_loss']
    
    def _close_position(self, position: dict, exit_candle: dict, reason: str) -> dict:
        exit_price = position['stop_loss'] if reason == 'stop' else exit_candle['close']
        position_size = position.get('position_size', 1)  # По умолчанию 1 лот
        
        if position['signal_type'] == SignalType.LONG:
            profit = (exit_price - position['entry_price']) * position_size  # ← ИСПРАВЛЕНО!
        else:
            profit = (position['entry_price'] - exit_price) * position_size  # ← ИСПРАВЛЕНО!

        
        return {
            'entry_price': position['entry_price'],
            'entry_time': position['entry_time'],      # ← ДОБАВЬТЕ
            'exit_price': exit_price,
            'exit_time': exit_candle['time'],          # ← ДОБАВЬТЕ
            'profit': profit,
            'reason': reason,
            'signal_type': position['signal_type']     # ← ДОБАВЬТЕ
        }


    def _calculate_metrics(self, trades: list, equity: list) -> dict:
        """Рассчитывает все метрики"""
        import numpy as np
        
        profits = [t['profit'] for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        # Drawdown
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        # Sharpe ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0
        
        return {
            'total_profit': sum(profits),
            'total_trades': len(trades),
            'win_rate': len(wins) / len(profits) if profits else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': abs(np.mean(losses)) if losses else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else 999,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe
        }
    
    def _generate_html_report(self, ticker: str, metrics: dict, trades: list, equity: list):
        """Генерирует HTML отчет с графиками"""
        Path("backtest_results").mkdir(exist_ok=True)
        
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{ticker} Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
}}
.container {{
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    border-radius: 15px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    padding: 30px;
}}
h1 {{
    color: #333;
    text-align: center;
    margin-bottom: 30px;
    font-size: 32px;
}}
.metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-bottom: 40px;
}}
.metric {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
}}
.metric strong {{
    display: block;
    font-size: 14px;
    margin-bottom: 10px;
    opacity: 0.9;
}}
.metric span {{
    display: block;
    font-size: 24px;
    font-weight: bold;
}}
.chart {{
    margin: 30px 0;
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
}}
</style></head><body>
<div class="container">
    <h1>📊 {ticker} Backtest Results</h1>
    
    <div class="metrics">
        <div class="metric">
            <strong>Total Profit</strong>
            <span>{metrics['total_profit']:.2f} ₽</span>
        </div>
        <div class="metric">
            <strong>Win Rate</strong>
            <span>{metrics['win_rate']*100:.1f}%</span>
        </div>
        <div class="metric">
            <strong>Profit Factor</strong>
            <span>{metrics['profit_factor']:.2f}</span>
        </div>
        <div class="metric">
            <strong>Max Drawdown</strong>
            <span>{metrics['max_drawdown']*100:.1f}%</span>
        </div>
        <div class="metric">
            <strong>Sharpe Ratio</strong>
            <span>{metrics['sharpe_ratio']:.2f}</span>
        </div>
        <div class="metric">
            <strong>Total Trades</strong>
            <span>{metrics['total_trades']}</span>
        </div>
    </div>
    
    <div id="equity" class="chart"></div>
    <div id="distribution" class="chart"></div>
</div>

<script>
var equityData = {{
    y: {equity},
    type: 'scatter',
    mode: 'lines',
    name: 'Equity',
    line: {{color: '#4CAF50', width: 3}},
    fill: 'tozeroy',
    fillcolor: 'rgba(76, 175, 80, 0.2)'
}};

var equityLayout = {{
    title: 'Equity Curve',
    xaxis: {{title: 'Trade Number'}},
    yaxis: {{title: 'Capital (₽)'}},
    hovermode: 'closest'
}};

Plotly.newPlot('equity', [equityData], equityLayout);

var profits = {[t['profit'] for t in trades]};
var distributionData = {{
    x: profits,
    type: 'histogram',
    marker: {{color: '#2196F3', line: {{color: '#1976D2', width: 1}}}},
    name: 'Trade P&L'
}};

var distributionLayout = {{
    title: 'Trade Distribution',
    xaxis: {{title: 'Profit (₽)'}},
    yaxis: {{title: 'Count'}},
    bargap: 0.05
}};

Plotly.newPlot('distribution', [distributionData], distributionLayout);
</script>
</body></html>"""
        
        filepath = f"backtest_results/{ticker}_report.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"\n📁 HTML отчет сохранен: {filepath}")


if __name__ == "__main__":
    # ЗАМЕНИТЕ НА ВАШ ТОКЕН!
    TOKEN = ""  # <<<< ЗАМЕНИТЕ!
    APP_NAME = "seykota_backtest"
    
    runner = RealBacktestRunner(TOKEN, APP_NAME)
    
    # Настройки стратегии
    strategy_settings = {
        "EMA_SHORT": "18",
        "EMA_LONG": "50",
        "MIN_CANDLES": "100",
        "KELLY_FRACTION": "0.25",
        "MAX_RISK": "0.05",
        "MIN_TRADES_FOR_KELLY": "30",
        "ATR_PERIOD": "14",
        "ATR_MULTIPLIER": "6.0",    # ← ИЗМЕНИТЕ
        "PERCENT_STOP": "0.07",      # ← ИЗМЕНИТЕ
        "SWING_PERIOD": "20"
    }
    
    # Запуск для VTBR
    results = runner.run(
        figi="BBG004730ZJ9",
        ticker="VTBR",
        strategy_settings=strategy_settings
    )

