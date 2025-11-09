# trade_backtesting/detailed_backtest_visualizer.py
# Полный оригинал, сохранён стиль, метрики и все графики
import pandas as pd
from pathlib import Path
from datetime import datetime

class DetailedBacktestVisualizer:
    @staticmethod
    def calculate_ema(prices, period):
        ema = [prices[0]]
        k = 2 / (period + 1)
        for price in prices[1:]:
            ema.append(price * k + ema[-1] * (1 - k))
        return ema

    @staticmethod
    def calculate_atr(candles_data, period=14):
        atr_values = [0]
        for i in range(1, len(candles_data)):
            high = candles_data[i]['high']
            low = candles_data[i]['low']
            prev_close = candles_data[i-1]['close']
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            if i < period:
                atr_values.append(sum([atr_values[j] if j == 0 else tr for j in range(i)]) / i)
            else:
                prev_atr = atr_values[-1]
                atr_values.append((prev_atr * (period - 1) + tr) / period)
        return atr_values

    @staticmethod
    def normalize_time(time_value):
        if isinstance(time_value, str):
            return pd.to_datetime(time_value)
        elif isinstance(time_value, (datetime, pd.Timestamp)):
            return pd.to_datetime(time_value)
        else:
            return time_value

    @staticmethod
    def generate_detailed_report(ticker: str, candles_data: list, trades: list,
                                equity: list, metrics: dict, kelly_history: list = None,
                                output_file: str = "backtest_report.html"):
        df = pd.DataFrame(candles_data)
        df['time_norm'] = df['time'].apply(DetailedBacktestVisualizer.normalize_time)
        df['time_str'] = df['time_norm'].dt.strftime('%Y-%m-%d %H:%M')
        ema18_values = DetailedBacktestVisualizer.calculate_ema(df['close'].tolist(), 18)
        ema50_values = DetailedBacktestVisualizer.calculate_ema(df['close'].tolist(), 50)
        atr_values = DetailedBacktestVisualizer.calculate_atr(candles_data, 14)
        entry_points = []
        exit_points = []
        for i, trade in enumerate(trades, 1):
            type_val = str(trade.get('signal_type', 'LONG'))
            if 'entry_time' in trade and trade['entry_time'] is not None:
                trade_entry_norm = DetailedBacktestVisualizer.normalize_time(trade['entry_time'])
                time_diffs = abs((df['time_norm'] - trade_entry_norm).dt.total_seconds())
                entry_position = time_diffs.argmin()
                entry_time_str = df.iloc[entry_position]['time_str']
            else:
                entry_position = max(0, int((i-1) * len(df) / max(1, len(trades))))
                entry_time_str = df.iloc[entry_position]['time_str']
            if 'exit_time' in trade and trade['exit_time'] is not None:
                trade_exit_norm = DetailedBacktestVisualizer.normalize_time(trade['exit_time'])
                time_diffs = abs((df['time_norm'] - trade_exit_norm).dt.total_seconds())
                exit_position = time_diffs.argmin()
                exit_time_str = df.iloc[exit_position]['time_str']
            else:
                exit_position = min(entry_position + 50, len(df) - 1)
                exit_time_str = df.iloc[exit_position]['time_str']
            entry_points.append({
                'time': entry_time_str,
                'y': float(trade['entry_price']),
                'trade_num': i,
                'type': type_val
            })
            exit_points.append({
                'time': exit_time_str,
                'y': float(trade['exit_price']),
                'trade_num': i,
                'profit': float(trade['profit']),
                'reason': trade.get('reason', 'unknown')
            })
        # ВИЗУАЛИЗАЦИЯ — ОРИГИНАЛ С ДОБАВЛЕННЫМИ МАРКЕРАМИ И ЛЕГЕНДОЙ
        html = f"""
<!DOCTYPE html><html><head>
<meta charset='utf-8'>
<title>{ticker} - Детальный Анализ</title>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }}
.container {{ max-width: 1600px; margin: 0 auto; background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }}
h1 {{ color: #333; text-align: center; margin-bottom: 30px; font-size: 36px; }}
.chart {{ margin: 30px 0; background: #f8f9fa; padding: 20px; border-radius: 10px; }}
.section-title {{ font-size: 24px; margin: 30px 0 15px 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
</style>
</head><body><div class='container'>
<h1>📊 {ticker} - Детальный Анализ</h1>

<div class='summary'>
    <div class='metric'>
        <strong>Total Profit</strong>
        <div class='value' style='color:{'color: #22c55e' if metrics['total_profit'] > 0 else 'color: #ef4444'}'>{metrics['total_profit']:.2f} ₽</div>
    </div>
    <div class='metric'><strong>Win Rate</strong><div class='value'>{metrics['win_rate']*100:.1f}%</div></div>
    <div class='metric'><strong>Profit Factor</strong><div class='value'>{metrics['profit_factor']:.2f}</div></div>
    <div class='metric'><strong>Max Drawdown</strong><div class='value'>{metrics['max_drawdown']*100:.1f}%</div></div>
    <div class='metric'><strong>Sharpe</strong><div class='value'>{metrics['sharpe_ratio']:.2f}</div></div>
    <div class='metric'><strong>Total Trades</strong><div class='value'>{metrics['total_trades']}</div></div>
</div>

<h2 class='section-title'>📈 График Цены с EMA(18), EMA(50) и Входами/Выходами</h2>
<div id='price-chart' class='chart'></div>
<h2 class='section-title'>📊 ATR(14)</h2>
<div id='atr-chart' class='chart'></div>
<h2 class='section-title'>💎 Kelly Criterion (%)</h2>
<div id='kelly-chart' class='chart'></div>
<h2 class='section-title'>💰 Кривая Капитала</h2>
<div id='equity-chart' class='chart'></div>
</div>
<script>
var priceData = {{ x: {df['time_str'].tolist()}, y: {df['close'].tolist()}, type: 'scatter', mode: 'lines', name: 'Цена', line: {{ color: '#3b82f6', width: 1.5 }} }};
var ema18Data = {{ x: {df['time_str'].tolist()}, y: {ema18_values}, type: 'scatter', mode: 'lines', name: 'EMA(18)', line: {{ color: '#22c55e', width: 2 }} }};
var ema50Data = {{ x: {df['time_str'].tolist()}, y: {ema50_values}, type: 'scatter', mode: 'lines', name: 'EMA(50)', line: {{ color: '#ef4444', width: 2 }} }};
var entryLongs = {{ x: {[p['time'] for p in entry_points if 'LONG' in p['type']]}, y: {[p['y'] for p in entry_points if 'LONG' in p['type']]}, mode: 'markers', name: '🟢 LONG Вход', marker: {{ size: 16, color: '#22c55e', symbol: 'triangle-up', line: {{ color: '#004d1a', width: 2 }} }}, text: {[f"Сделка #{p['trade_num']}" for p in entry_points if 'LONG' in p['type']]}};
var entryShorts = {{ x: {[p['time'] for p in entry_points if 'SHORT' in p['type']]}, y: {[p['y'] for p in entry_points if 'SHORT' in p['type']]}, mode: 'markers', name: '🔴 SHORT Вход', marker: {{ size: 16, color: '#ef4444', symbol: 'triangle-down', line: {{ color: '#330000', width: 2 }} }}, text: {[f"Сделка #{p['trade_num']}" for p in entry_points if 'SHORT' in p['type']]}};
var stopMarkers = {{x: {[p['time'] for p in exit_points if p['reason']=='stop']},y: {[p['y'] for p in exit_points if p['reason']=='stop']},mode: 'markers',name: '✖️ СТОП',marker: {{size: 16, color: '#333', symbol: 'cross', line: {{ color: 'yellow', width: 3 }} }},text: {[f"Сделка #{p['trade_num']}" for p in exit_points if p['reason']=='stop']}};
var endMarkers = {{x: {[p['time'] for p in exit_points if p['reason']=='end']},y: {[p['y'] for p in exit_points if p['reason']=='end']},mode: 'markers',name: '🔵 КОНЕЦ',marker: {{size: 12, color: '#0047ab', symbol: 'square', line: {{ color: 'white', width: 2 }} }},text: {[f"Сделка #{p['trade_num']}" for p in exit_points if p['reason']=='end']}};
Plotly.newPlot('price-chart', [priceData, ema18Data, ema50Data, entryLongs, entryShorts, stopMarkers, endMarkers], { title: 'Цена, EMA(18), EMA(50) с входами/выходами', xaxis: { title: 'Дата и время'}, yaxis: { title: 'Цена (₽)'}, hovermode: 'closest'});
Plotly.newPlot('atr-chart', [{x: {df['time_str'].tolist()}, y: {atr_values}, type: 'scatter', mode: 'lines', name: 'ATR(14)', line: {color: '#6c2eb6', width: 2}}], {title: 'ATR(14)', xaxis: {title: 'Дата и время'}, yaxis: {title: 'ATR'}});
var kellyValues = { [k * 100 for k in kelly_history] if kelly_history else [] };
Plotly.newPlot('kelly-chart', [{y: kellyValues,type: 'scatter',mode: 'lines+markers',line: { color: '#8b5cf6', width: 2 },marker: { size: 6, color: '#8b5cf6' },fill: 'tozeroy',fillcolor: 'rgba(139, 92, 246, 0.2)'}], {title: 'Kelly % по сделкам', xaxis: { title: 'Сделка #' },yaxis: { title: 'Kelly %' },height: 300});
Plotly.newPlot('equity-chart', [{y: {equity},type: 'scatter',mode: 'lines+markers',name: 'Equity',line: {color: '#4CAF50', width: 3},fill: 'tozeroy',fillcolor: 'rgba(76, 175, 80, 0.2)' }], {title: 'Equity Curve', xaxis: {title: 'Trade Number'},yaxis: {title: 'Capital (₽)'},hovermode: 'closest'});
</script></body></html>
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
