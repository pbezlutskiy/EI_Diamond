# Путь: invest-bot/trade_system/strategies/seykota_kelly_strategy.py
"""
Стратегия Эда Сейкоты с формулой Келли (исправлена логика swing, расчёта percent stop, и безопасности stop loss)
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
from trade_system.signal import Signal, SignalType

logger = logging.getLogger(__name__)

class SeykoaKellyStrategy:
    """Трендовая стратегия с управлением капиталом по Келли"""

    def __init__(self, figi: str, ticker: str, settings: Dict[str, Any]):
        self.figi = figi
        self.ticker = ticker
        self.ema_short_period = int(settings.get("EMA_SHORT", "18"))
        self.ema_long_period = int(settings.get("EMA_LONG", "50"))
        self.min_candles = int(settings.get("MIN_CANDLES", "100"))
        self.atr_period = int(settings.get("ATR_PERIOD", "14"))
        self.atr_multiplier = float(settings.get("ATR_MULTIPLIER", "2.0"))
        self.min_atr_distance = float(settings.get("MIN_ATR_DISTANCE", "3.0"))
        # percent_stop задаётся как 0.07 для 7%
        self.percent_stop = float(settings.get("PERCENT_STOP", 0.07))  # Исправлено!
        self.swing_period = int(settings.get("SWING_PERIOD", "40"))
        self.kelly_fraction = float(settings.get("KELLY_FRACTION", "0.25"))
        self.max_risk = float(settings.get("MAX_RISK", "0.05"))
        self.min_trades_for_kelly = int(settings.get("MIN_TRADES_FOR_KELLY", "30"))
        self.trade_history = []
        logger.info(f"SeykoaKellyStrategy initialized for {ticker}")
        logger.info(f"  EMA periods: {self.ema_short_period}/{self.ema_long_period}")
        logger.info(f"  ATR: period={self.atr_period}, multiplier={self.atr_multiplier}, min_distance={self.min_atr_distance}")
        logger.info(f"  Swing period: {self.swing_period}")

    @staticmethod
    def _to_float(value) -> float:
        if hasattr(value, 'units') and hasattr(value, 'nano'):
            return float(value.units) + float(value.nano) / 1e9
        return float(value)

    def analyze_candles(self, candles: list) -> Optional[Signal]:
        if len(candles) < self.min_candles:
            return None
        closes = [self._to_float(c.close) for c in candles]
        ema_short = self._calculate_ema(closes, self.ema_short_period)
        ema_long = self._calculate_ema(closes, self.ema_long_period)
        atr = self._calculate_atr(candles, self.atr_period)
        current_ema_short = ema_short[-1]
        current_ema_long = ema_long[-1]
        prev_ema_short = ema_short[-2]
        prev_ema_long = ema_long[-2]
        current_price = closes[-1]
        current_atr = atr[-1]
        signal_type = None
        if prev_ema_short <= prev_ema_long and current_ema_short > current_ema_long:
            signal_type = SignalType.LONG
        elif prev_ema_short >= prev_ema_long and current_ema_short < current_ema_long:
            signal_type = SignalType.SHORT
        if signal_type is None:
            return None
        swing_level = self._calculate_swing_level(candles, signal_type)
        stop_loss = self._calculate_stop_loss(
            signal_type, 
            current_price, 
            current_atr, 
            swing_level
        )
        return Signal(
            signal_type=signal_type,
            figi=self.figi,
            stop_loss_level=Decimal(str(stop_loss)),
                        take_profit_level=Decimal(0)
        )

    def _calculate_ema(self, prices: list, period: int) -> list:
        ema = [prices[0]]
        k = 2 / (period + 1)
        for price in prices[1:]:
            ema_value = price * k + ema[-1] * (1 - k)
            ema.append(ema_value)
        return ema

    def _calculate_atr(self, candles: list, period: int) -> list:
        atr_values = [0]
        for i in range(1, len(candles)):
            high = self._to_float(candles[i].high)
            low = self._to_float(candles[i].low)
            prev_close = self._to_float(candles[i-1].close)
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

    def _calculate_swing_level(self, candles: list, signal_type: SignalType) -> float:
        recent_candles = candles[-self.swing_period:]
        if signal_type == SignalType.LONG:
            swing = min(self._to_float(c.low) for c in recent_candles)
        else:
            swing = max(self._to_float(c.high) for c in recent_candles)
        logger.info(f"SWING_LEVEL for {signal_type.name}: {swing:.2f} (из последних {self.swing_period} свечей)")
        return swing

    def _calculate_stop_loss(self, signal_type: SignalType, entry_price: float, atr: float, swing_level: float) -> float:
        if signal_type == SignalType.LONG:
            atr_stop = entry_price - (atr * self.atr_multiplier)
            percent_stop = entry_price * (1 - self.percent_stop)
            swing_stop = swing_level
            min_distance_stop = entry_price - (atr * self.min_atr_distance)
            # Используем только СМЫСЛОВЫЕ уровни stop: те, которые ниже entry_price
            all_stops = [val for val in [atr_stop, percent_stop, swing_stop, min_distance_stop] if val < entry_price]
            hybrid_stop = min(all_stops) if all_stops else min(atr_stop, percent_stop, swing_stop, min_distance_stop)
            logger.info(f"LONG STOP: ATR={atr_stop:.2f}, Percent={percent_stop:.2f}, Swing={swing_stop:.2f}, MinDist={min_distance_stop:.2f}, HYBRID={hybrid_stop:.2f}")
            return hybrid_stop
        else:  # SHORT
            atr_stop = entry_price + (atr * self.atr_multiplier)
            percent_stop = entry_price * (1 + self.percent_stop)
            swing_stop = swing_level
            min_distance_stop = entry_price + (atr * self.min_atr_distance)
            # Используем только СМЫСЛОВЫЕ уровни stop: те, которые выше entry_price
            all_stops = [val for val in [atr_stop, percent_stop, swing_stop, min_distance_stop] if val > entry_price]
            hybrid_stop = max(all_stops) if all_stops else max(atr_stop, percent_stop, swing_stop, min_distance_stop)
            logger.info(f"SHORT STOP: ATR={atr_stop:.2f}, Percent={percent_stop:.2f}, Swing={swing_stop:.2f}, MinDist={min_distance_stop:.2f}, HYBRID={hybrid_stop:.2f}")
            return hybrid_stop

    def record_trade_result(self, signal_type: SignalType, entry_price: float, exit_price: float, entry_time: datetime, exit_time: datetime):
        if signal_type == SignalType.LONG:
            profit = exit_price - entry_price
        else:
            profit = entry_price - exit_price
        self.trade_history.append({
            'signal_type': signal_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'win': profit > 0
        })

    def _kelly_calculator(self) -> float:
        if len(self.trade_history) < self.min_trades_for_kelly:
            return self.kelly_fraction
        wins = [t for t in self.trade_history if t['win']]
        losses = [t for t in self.trade_history if not t['win']]
        if not wins or not losses:
            return self.kelly_fraction
        win_rate = len(wins) / len(self.trade_history)
        avg_win = sum(t['profit'] for t in wins) / len(wins)
        avg_loss = abs(sum(t['profit'] for t in losses) / len(losses))
        if avg_win == 0:
            return self.kelly_fraction
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_pct = min(max(kelly_pct, 0), self.max_risk)
        return kelly_pct * self.kelly_fraction
