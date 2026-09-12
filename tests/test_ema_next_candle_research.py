"""Tests for EMA next-candle research."""

import unittest

from candles import Candle
from ema_next_candle_research import PERIODS, QUANTILES, atr14, research


class EmaNextCandleResearchTests(unittest.TestCase):
    def test_atr_is_positive_and_matches_input_length(self):
        candles = _candles(120)
        values = atr14(candles)
        self.assertEqual(len(values), len(candles))
        self.assertTrue(all(value > 0 for value in values))

    def test_research_covers_every_period_and_quantile(self):
        rows = research(_candles(300))
        self.assertEqual(len(rows), len(PERIODS) * len(QUANTILES))
        self.assertEqual({row.period for row in rows}, set(PERIODS))
        self.assertEqual({row.quantile for row in rows}, set(QUANTILES))

    def test_thresholds_are_selected_from_training_half_only(self):
        candles = _candles(300)
        original = research(candles)
        changed = list(candles)
        for index in range(150, 300):
            candle = changed[index]
            changed[index] = Candle(
                candle.timestamp,
                candle.open * 1.2,
                candle.high * 1.2,
                candle.low * 1.2,
                candle.close * 1.2,
            )
        modified = research(changed)
        self.assertEqual(
            [row.threshold_atr for row in original],
            [row.threshold_atr for row in modified],
        )


def _candles(count):
    candles = []
    price = 100.0
    for index in range(count):
        price += (index % 7 - 3) * 0.1
        candles.append(Candle(index * 300, price, price + 1.0, price - 1.0, price))
    return candles


if __name__ == "__main__":
    unittest.main()
