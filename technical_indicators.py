"""OHLCV indicator calculations used by the multi-timeframe technical chart.

The formulas are independent Python implementations inspired by the three user-
provided Pine indicators. They do not execute or embed Pine code.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


TIMEFRAME_SOURCE = {
    "1M": ("1m", "7d"), "5M": ("5m", "60d"),
    "15M": ("15m", "60d"), "30M": ("30m", "60d"),
    "1H": ("60m", "730d"), "4H": ("60m", "730d"),
    "1D": ("1d", "5y"), "1W": ("1d", "max"),
    "1MO": ("1d", "max"), "1Y": ("1d", "max"),
}


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate completed source bars without mixing separate US sessions."""
    data = frame[["Open", "High", "Low", "Close", "Volume"]].copy().sort_index()
    if timeframe not in {"4H", "1W", "1MO", "1Y"}:
        return data
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if timeframe == "4H":
        minutes = data.index.hour * 60 + data.index.minute
        bucket = np.maximum((minutes - 570) // 240, 0)
        groups = [data.index.normalize(), bucket]
        result = data.groupby(groups, sort=True).agg(agg)
        # Keep a chronological timestamp for each intraday bar.
        result.index = pd.DatetimeIndex([
            day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=int(part * 4))
            for day, part in result.index
        ])
    else:
        frequency = {"1W": "W-FRI", "1MO": "ME", "1Y": "YE"}[timeframe]
        result = data.resample(frequency).agg(agg)
    return result.dropna(subset=["Open", "High", "Low", "Close"])


def weighted_ma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length, min_periods=length).apply(
        lambda values: float(np.dot(values, weights) / weights.sum()), raw=True)


def moving_average(series: pd.Series, length: int, kind: str, volume: pd.Series | None = None,
                   alma_offset: float = .85, alma_sigma: float = 6.0,
                   jma_phase: int = 3, jma_power: int = 1) -> pd.Series:
    length = max(1, int(length))
    source = pd.to_numeric(series, errors="coerce").astype(float)
    kind = kind.upper()
    if kind == "SMA":
        return source.rolling(length, min_periods=length).mean()
    if kind == "EMA":
        return source.ewm(span=length, adjust=False, min_periods=length).mean()
    if kind == "WMA":
        return weighted_ma(source, length)
    if kind == "SWMA":
        weights = np.array([1, 2, 2, 1], dtype=float) / 6
        return source.rolling(4, min_periods=4).apply(
            lambda values: float(np.dot(values, weights)), raw=True)
    if kind == "HMA":
        half = weighted_ma(source, max(1, round(length / 2)))
        full = weighted_ma(source, length)
        return weighted_ma(2 * half - full, max(1, round(math.sqrt(length))))
    if kind == "VWMA":
        if volume is None:
            return pd.Series(np.nan, index=source.index)
        vol = pd.to_numeric(volume, errors="coerce").clip(lower=0).fillna(0)
        total = vol.rolling(length, min_periods=length).sum().replace(0, np.nan)
        return source.mul(vol).rolling(length, min_periods=length).sum().div(total)
    if kind == "DEMA":
        ema = source.ewm(span=length, adjust=False, min_periods=length).mean()
        return 2 * ema - ema.ewm(span=length, adjust=False, min_periods=length).mean()
    if kind == "TEMA":
        e1 = source.ewm(span=length, adjust=False, min_periods=length).mean()
        e2 = e1.ewm(span=length, adjust=False, min_periods=length).mean()
        e3 = e2.ewm(span=length, adjust=False, min_periods=length).mean()
        return 3 * e1 - 3 * e2 + e3
    if kind == "ZLEMA":
        lag = (length - 1) // 2
        adjusted = 2 * source - source.shift(lag)
        return adjusted.ewm(span=length, adjust=False, min_periods=length).mean()
    if kind == "ALMA":
        position = alma_offset * (length - 1)
        scale = length / max(alma_sigma, .01)
        weights = np.exp(-((np.arange(length) - position) ** 2) / (2 * scale ** 2))
        weights /= weights.sum()
        return source.rolling(length, min_periods=length).apply(
            lambda values: float(np.dot(values, weights)), raw=True)
    if kind == "JMA":
        beta = .45 * (length - 1) / (.45 * (length - 1) + 2)
        alpha = beta ** max(1, jma_power)
        phase = min(2.5, max(.5, jma_phase / 100 + 1.5))
        output = np.full(len(source), np.nan)
        e0 = e1 = e2 = jma = 0.0
        for i, value in enumerate(source.to_numpy()):
            if not np.isfinite(value):
                continue
            if i == 0:
                e0 = jma = value
            e0 = (1 - alpha) * value + alpha * e0
            e1 = (value - e0) * (1 - beta) + beta * e1
            e2 = (e0 + phase * e1 - jma) * (1 - alpha) ** 2 + alpha ** 2 * e2
            jma += e2
            if i >= length - 1:
                output[i] = jma
        return pd.Series(output, index=source.index)
    raise ValueError(f"Unsupported MA type: {kind}")


def bollinger_bands(close: pd.Series, length: int, multiplier: float) -> pd.DataFrame:
    if not isinstance(close, pd.Series):
        raise TypeError("bollinger_bands expects a Close price Series, not an OHLCV frame")
    mid = close.rolling(length, min_periods=length).mean()
    deviation = close.rolling(length, min_periods=length).std(ddof=0) * multiplier
    return pd.DataFrame({"mid": mid, "upper": mid + deviation, "lower": mid - deviation})


def anchored_vwap(frame: pd.DataFrame, anchor: str = "Week") -> pd.DataFrame:
    """Volume-weighted HLC3 and weighted standard deviation by calendar anchor."""
    idx = frame.index
    if anchor == "All":
        groups = np.zeros(len(idx), dtype=int)
    elif anchor == "Session":
        groups = idx.normalize()
    elif anchor == "Week":
        groups = idx.to_period("W-FRI")
    elif anchor == "Month":
        groups = idx.to_period("M")
    elif anchor == "Quarter":
        groups = idx.to_period("Q")
    elif anchor == "Year":
        groups = idx.to_period("Y")
    else:
        raise ValueError(f"Unsupported VWAP anchor: {anchor}")
    typical = frame[["High", "Low", "Close"]].mean(axis=1)
    volume = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0).clip(lower=0)
    cumulative_volume = volume.groupby(groups).cumsum().replace(0, np.nan)
    vwap = typical.mul(volume).groupby(groups).cumsum().div(cumulative_volume)
    second = typical.pow(2).mul(volume).groupby(groups).cumsum().div(cumulative_volume)
    std = np.sqrt((second - vwap.pow(2)).clip(lower=0))
    return pd.DataFrame({"vwap": vwap, "std": std})


def adx_di(frame: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    high, low, close = frame["High"], frame["Low"], frame["Close"]
    upward = high.diff()
    downward = -low.diff()
    plus_dm = pd.Series(np.where((upward > downward) & (upward > 0), upward, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((downward > upward) & (downward > 0), downward, 0.0), index=frame.index)
    true_range = pd.concat([high - low, (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1).max(axis=1)
    smooth = lambda series: series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    atr = smooth(true_range).replace(0, np.nan)
    plus_di = 100 * smooth(plus_dm) / atr
    minus_di = 100 * smooth(minus_dm) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"ADX": smooth(dx), "+DI": plus_di, "-DI": minus_di})


def ssl_hybrid(frame: pd.DataFrame, baseline_length: int = 60, baseline_type: str = "HMA",
               ssl2_length: int = 5, ssl2_type: str = "JMA", exit_length: int = 15,
               exit_type: str = "HMA", channel_multiplier: float = .2,
               atr_length: int = 14, jma_phase: int = 3, jma_power: int = 1,
               atr_criterion: float = .9) -> pd.DataFrame:
    def ma(series: pd.Series, length: int, kind: str) -> pd.Series:
        return moving_average(series, length, kind, frame["Volume"],
                              jma_phase=jma_phase, jma_power=jma_power)

    close, high, low = frame["Close"], frame["High"], frame["Low"]
    baseline = ma(close, baseline_length, baseline_type)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    channel_width = tr.ewm(span=baseline_length, adjust=False).mean() * channel_multiplier

    def switched(length: int, kind: str) -> pd.Series:
        upper, lower = ma(high, length, kind), ma(low, length, kind)
        side = pd.Series(np.where(close > upper, 1, np.where(close < lower, -1, np.nan)),
                         index=frame.index).ffill().fillna(0)
        return pd.Series(np.where(side < 0, upper, lower), index=frame.index)

    atr = weighted_ma(tr, atr_length)
    ssl1 = switched(baseline_length, baseline_type)
    ssl2 = switched(ssl2_length, ssl2_type)
    exit_line = switched(exit_length, exit_type)
    buy_cont = (close > baseline) & (close > ssl2) & (close - atr * atr_criterion < ssl2)
    sell_cont = (close < baseline) & (close < ssl2) & (close + atr * atr_criterion > ssl2)
    return pd.DataFrame({
        "baseline": baseline, "channel_upper": baseline + channel_width,
        "channel_lower": baseline - channel_width,
        "ssl1": ssl1, "ssl2": ssl2, "exit": exit_line, "atr": atr,
        "buy_cont": buy_cont & ~buy_cont.shift(fill_value=False),
        "sell_cont": sell_cont & ~sell_cont.shift(fill_value=False),
        "exit_long": (close < exit_line) & (close.shift() >= exit_line.shift()),
        "exit_short": (close > exit_line) & (close.shift() <= exit_line.shift()),
    })


def heikin_trend(frame: pd.DataFrame, length: int = 9, ma_type: str = "EMA",
                 alma_offset: float = .85, alma_sigma: float = 6.0) -> pd.DataFrame:
    """Smooth Heikin-Ashi OHLC and compute the source script's trend score."""
    ha_close = frame[["Open", "High", "Low", "Close"]].mean(axis=1)
    ha_open_values = np.empty(len(frame), dtype=float)
    for i, (real_open, synthetic_close) in enumerate(zip(frame["Open"], ha_close)):
        ha_open_values[i] = ((real_open + synthetic_close) / 2 if i == 0 else
                             (ha_open_values[i - 1] + ha_close.iloc[i - 1]) / 2)
    ha_open = pd.Series(ha_open_values, index=frame.index)
    ha_high = pd.concat([frame["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([frame["Low"], ha_open, ha_close], axis=1).min(axis=1)
    smooth = lambda series: moving_average(series, length, ma_type, frame["Volume"],
                                           alma_offset=alma_offset, alma_sigma=alma_sigma)
    result = pd.DataFrame({"open": smooth(ha_open), "close": smooth(ha_close),
                           "high": smooth(ha_high), "low": smooth(ha_low)})
    result["score"] = 100 * (result["close"] - result["open"]) / (
        result["high"] - result["low"]).replace(0, np.nan)
    return result


def candle_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    """Simple confirmed OHLC patterns; no forward-looking bars or trade signals."""
    opening, high, low, close = (frame[field] for field in ("Open", "High", "Low", "Close"))
    body = (close - opening).abs()
    span = (high - low).replace(0, np.nan)
    prior_open, prior_close = opening.shift(), close.shift()
    bullish_engulfing = ((prior_close < prior_open) & (close > opening)
                         & (opening <= prior_close) & (close >= prior_open))
    bearish_engulfing = ((prior_close > prior_open) & (close < opening)
                         & (opening >= prior_close) & (close <= prior_open))
    lower_shadow = opening.combine(close, min) - low
    upper_shadow = high - opening.combine(close, max)
    hammer = (lower_shadow >= 2 * body) & (upper_shadow <= body) & (body / span > .05)
    shooting_star = (upper_shadow >= 2 * body) & (lower_shadow <= body) & (body / span > .05)
    doji = body / span <= .1
    return pd.DataFrame({"Bullish Engulfing": bullish_engulfing.fillna(False),
                         "Bearish Engulfing": bearish_engulfing.fillna(False),
                         "Hammer": hammer.fillna(False),
                         "Shooting Star": shooting_star.fillna(False),
                         "Doji": doji.fillna(False)}, index=frame.index)
