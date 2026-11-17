import unittest

import numpy as np
import pandas as pd

from technical_indicators import (TIMEFRAME_SOURCE, adx_di, anchored_vwap,
    bollinger_bands, candle_patterns, heikin_trend, moving_average,
    resample_ohlcv, ssl_hybrid)


def fixture_bars(count=260):
    index = pd.bdate_range("2025-01-01", periods=count)
    close = pd.Series(100 + np.arange(count) * .2 + np.sin(np.arange(count) / 9), index=index)
    return pd.DataFrame({"Open": close - .3, "High": close + 1,
                         "Low": close - 1, "Close": close,
                         "Volume": np.full(count, 1000)}, index=index)


class TechnicalIndicatorTests(unittest.TestCase):
    def test_all_requested_timeframes_have_sources(self):
        self.assertEqual(set(TIMEFRAME_SOURCE),
            {"1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W", "1MO", "1Y"})

    def test_bollinger_and_vwap_with_constant_prices(self):
        bars = fixture_bars(30)
        bars[["Open", "High", "Low", "Close"]] = 100.0
        bands = bollinger_bands(bars["Close"], 20, 2)
        vwap = anchored_vwap(bars, "Week")
        self.assertAlmostEqual(float(bands["upper"].iloc[-1]), 100)
        self.assertAlmostEqual(float(bands["lower"].iloc[-1]), 100)
        self.assertAlmostEqual(float(vwap["vwap"].iloc[-1]), 100)
        self.assertAlmostEqual(float(vwap["std"].iloc[-1]), 0)
        with self.assertRaisesRegex(TypeError, "Close price Series"):
            bollinger_bands(bars, 20, 2)

    def test_ssl_and_adx_have_final_values(self):
        bars = fixture_bars()
        ssl = ssl_hybrid(bars)
        directional = adx_di(bars)
        self.assertTrue(np.isfinite(ssl["baseline"].iloc[-1]))
        self.assertTrue(np.isfinite(ssl["ssl2"].iloc[-1]))
        self.assertTrue(all(ssl[field].dtype == bool for field in
            ("buy_cont", "sell_cont", "exit_long", "exit_short")))
        self.assertTrue(np.isfinite(directional["ADX"].iloc[-1]))
        self.assertGreater(float(directional["+DI"].iloc[-1]), 0)

    def test_heikin_trend_and_alma(self):
        bars = fixture_bars()
        self.assertTrue(np.isfinite(heikin_trend(bars)["score"].iloc[-1]))
        self.assertTrue(np.isfinite(moving_average(bars["Close"], 9, "ALMA").iloc[-1]))
        self.assertTrue(np.isfinite(moving_average(bars["Close"], 9, "SWMA").iloc[-1]))

    def test_yearly_aggregation_and_candle_patterns(self):
        bars = fixture_bars()
        yearly = resample_ohlcv(bars, "1Y")
        self.assertEqual(float(yearly["Open"].iloc[0]), float(bars["Open"].iloc[0]))
        self.assertEqual(float(yearly["Close"].iloc[-1]), float(bars["Close"].iloc[-1]))
        patterns = candle_patterns(bars)
        self.assertEqual(len(patterns), len(bars))
        self.assertTrue(all(patterns[column].dtype == bool for column in patterns))

    def test_four_hour_bars_do_not_merge_trading_days(self):
        index = pd.DatetimeIndex(["2026-10-12 09:30", "2026-10-12 13:30",
                                  "2026-10-13 09:30", "2026-10-13 13:30"])
        bars = pd.DataFrame({"Open": [1, 2, 3, 4], "High": [2, 3, 4, 5],
                             "Low": [0, 1, 2, 3], "Close": [1, 2, 3, 4],
                             "Volume": [10, 20, 30, 40]}, index=index)
        result = resample_ohlcv(bars, "4H")
        self.assertEqual(len(result), 4)
        self.assertEqual(float(result["Volume"].sum()), 100)


if __name__ == "__main__":
    unittest.main()
