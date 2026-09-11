"""Compare EMA5/8/13/26 signals for the next BTC five-minute candle.

The first half of the history selects fixed distance thresholds; the second half
is untouched holdout data.  A plain EMA trend rule is tested alongside the
opposite, short-horizon mean-reversion rule.  Distance is normalized by ATR14 so
one threshold remains meaningful across BTC price and volatility regimes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

from backtest import load_candles_csv
from btc5m_strategy import Candle
from divergence_stars import ema

PERIODS = (5, 8, 13, 26)
QUANTILES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975)


@dataclass(frozen=True)
class EmaResult:
    rule: str
    period: int
    quantile: float
    threshold_atr: float
    train_count: int
    train_win_rate_pct: float
    train_average_return_pct: float
    holdout_count: int
    holdout_win_rate_pct: float
    holdout_average_return_pct: float


def atr14(candles: Sequence[Candle]) -> list[float]:
    if not candles:
        return []
    ranges = []
    for index, candle in enumerate(candles):
        previous = candles[index - 1].close if index else candle.open
        ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous),
                abs(candle.low - previous),
            )
        )
    output = [ranges[0]]
    for value in ranges[1:]:
        output.append((output[-1] * 13.0 + value) / 14.0)
    return output


def _summary(values: Sequence[float]) -> tuple[int, float, float]:
    return (
        len(values),
        sum(value > 0 for value in values) / len(values) * 100.0 if values else 0.0,
        fmean(values) if values else 0.0,
    )


def research(candles: Sequence[Candle]) -> list[EmaResult]:
    if len(candles) < 100:
        raise ValueError("at least 100 candles are required")
    closes = [candle.close for candle in candles]
    averages = {period: ema(closes, period) for period in PERIODS}
    atr = atr14(candles)
    split = len(candles) // 2
    results: list[EmaResult] = []

    for period in PERIODS:
        distances = sorted(
            abs(closes[index] - averages[period][index]) / atr[index]
            for index in range(30, split)
            if atr[index] > 0
        )
        for quantile in QUANTILES:
            threshold = distances[int((len(distances) - 1) * quantile)]
            samples: list[list[float]] = [[], []]
            for index in range(30, len(candles) - 1):
                if abs(closes[index] - averages[period][index]) / atr[index] < threshold:
                    continue
                # Five-minute BTC has slight one-bar mean reversion: after an
                # extreme move above an EMA predict DOWN, and vice versa.
                direction = -1.0 if closes[index] > averages[period][index] else 1.0
                value = (closes[index + 1] / closes[index] - 1.0) * 100.0 * direction
                samples[0 if index < split else 1].append(value)
            train = _summary(samples[0])
            holdout = _summary(samples[1])
            results.append(
                EmaResult(
                    rule="extreme_mean_reversion",
                    period=period,
                    quantile=quantile,
                    threshold_atr=threshold,
                    train_count=train[0],
                    train_win_rate_pct=train[1],
                    train_average_return_pct=train[2],
                    holdout_count=holdout[0],
                    holdout_win_rate_pct=holdout[1],
                    holdout_average_return_pct=holdout[2],
                )
            )
    return results


def robust_rank(row: EmaResult) -> tuple[float, float, float]:
    return (
        min(row.train_win_rate_pct, row.holdout_win_rate_pct),
        row.holdout_win_rate_pct,
        row.holdout_average_return_pct,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research EMA next-candle accuracy.")
    parser.add_argument(
        "--candles", type=Path, default=Path("data/coinbase_btcusd_5m_180d.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/ema_next_candle.json")
    )
    args = parser.parse_args()

    rows = research(load_candles_csv(args.candles))
    ranked = sorted(rows, key=robust_rank, reverse=True)
    print("EMA  quantile threshold  train n/win/avg       holdout n/win/avg")
    for row in ranked[:12]:
        print(
            f"{row.period:>3}  {row.quantile:>7.3f}  {row.threshold_atr:>7.3f}ATR  "
            f"{row.train_count:>5}/{row.train_win_rate_pct:>5.1f}%/"
            f"{row.train_average_return_pct:>+7.4f}%  "
            f"{row.holdout_count:>5}/{row.holdout_win_rate_pct:>5.1f}%/"
            f"{row.holdout_average_return_pct:>+7.4f}%"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(
            {"best": asdict(ranked[0]), "rows": [asdict(row) for row in rows]},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
