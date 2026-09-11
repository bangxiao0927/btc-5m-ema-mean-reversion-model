"""Non-repainting MACD divergence stars for closed candles.

The rules follow the common 通达信 star formula: a bar qualifies only when it is
the highest high (or lowest low) of the last ``lookback`` bars, and MACD ``DIFF``
disagrees with the previous extreme inside that lookback.  ``ST`` is the 通达信
MACD bar, ``(DIFF - DEA) * 2``.

Every input is taken from the current bar or earlier bars, so a marker never
changes once its candle closes.  The original formula compares ``ST`` with the
constant ``0.5``, which only makes sense for a ten-unit share price, so the
threshold here scales with price instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from btc5m_strategy import Candle

BEARISH_LABELS = {1: "背离1星", 2: "背离2星", 3: "背离3星"}
BULLISH_LABELS = {1: "背离1星", 2: "背离2星", 3: "背离3星"}
REVERSAL_DOWN_LABEL = "顶转折"
REVERSAL_UP_LABEL = "底转折"


@dataclass(frozen=True)
class DivergenceStar:
    timestamp: int
    side: str
    stars: int
    label: str
    price: float
    reference_timestamp: int
    reference_price: float
    diff: float
    reference_diff: float
    dea: float
    reference_dea: float
    st: float
    reference_st: float
    ema5: float
    reference_ema5: float
    ema8: float
    reference_ema8: float
    ema13: float
    reference_ema13: float
    ema26: float
    reference_ema26: float


def ema(values: Sequence[float], period: int) -> list[float]:
    """通达信 EMA: seeded with the first value, then the usual recursion."""

    if period < 1:
        raise ValueError("period must be positive")
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    output = [float(values[0])]
    for value in values[1:]:
        output.append(alpha * float(value) + (1.0 - alpha) * output[-1])
    return output


def macd_series(
    closes: Sequence[float], short: int = 12, long: int = 26, signal: int = 9
) -> tuple[list[float], list[float], list[float]]:
    """Return ``DIFF``, ``DEA`` and ``ST`` where ``ST = (DIFF - DEA) * 2``."""

    fast = ema(closes, short)
    slow = ema(closes, long)
    diff = [fast[index] - slow[index] for index in range(len(closes))]
    dea = ema(diff, signal)
    st = [(diff[index] - dea[index]) * 2.0 for index in range(len(closes))]
    return diff, dea, st


def previous_extreme_offset(
    values: Sequence[float], index: int, lookback: int, highest: bool
) -> int | None:
    """通达信 ``HHVBARS(REF(H,1),lookback-1)+1`` as a positive bar offset."""

    start = index - lookback
    if start < 0 or index < 1:
        return None
    best_offset: int | None = None
    best_value: float | None = None
    # Scan from the nearest previous bar so ties keep the most recent extreme.
    for offset in range(1, lookback + 1):
        value = values[index - offset]
        if (
            best_value is None
            or (highest and value > best_value)
            or (not highest and value < best_value)
        ):
            best_value = value
            best_offset = offset
    return best_offset


def divergence_stars(
    candles: Sequence[Candle],
    lookback: int = 60,
    st_threshold_ratio: float = 0.0004,
    include_last: bool = False,
) -> list[DivergenceStar]:
    """Grade every closed candle that carries a MACD divergence marker.

    ``include_last`` keeps the newest candle, which is still forming on a live
    chart and may therefore change until it closes.
    """

    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if st_threshold_ratio < 0:
        raise ValueError("st_threshold_ratio cannot be negative")
    if len(candles) < lookback + 1:
        return []

    closes = [candle.close for candle in candles]
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]
    diff, dea, st = macd_series(closes)
    ema5 = ema(closes, 5)
    ema8 = ema(closes, 8)
    ema13 = ema(closes, 13)
    ema26 = ema(closes, 26)
    last_index = len(candles) - 1 if include_last else len(candles) - 2
    markers: list[DivergenceStar] = []

    for index in range(lookback, last_index + 1):
        threshold = st_threshold_ratio * closes[index]
        current_st = st[index]
        previous_st = st[index - 1]

        if highs[index] >= max(highs[index - lookback + 1 : index + 1]):
            offset = previous_extreme_offset(highs, index, lookback - 1, True)
            if offset is not None:
                reference = index - offset
                stars = None
                if diff[reference] > diff[index]:
                    if st[reference] <= current_st and current_st > 0:
                        stars = 1
                    elif st[reference] > current_st and current_st > 0:
                        stars = 2
                    elif st[reference] > current_st and current_st <= 0:
                        stars = 3
                    label = BEARISH_LABELS.get(stars or 0)
                elif previous_st > 0 and current_st < -threshold:
                    stars = 4
                    label = REVERSAL_DOWN_LABEL
                else:
                    label = None
                if stars is not None and label is not None:
                    markers.append(
                        DivergenceStar(
                            timestamp=candles[index].timestamp,
                            side="bearish",
                            stars=stars,
                            label=label,
                            price=highs[index],
                            reference_timestamp=candles[reference].timestamp,
                            reference_price=highs[reference],
                            diff=diff[index],
                            reference_diff=diff[reference],
                            dea=dea[index],
                            reference_dea=dea[reference],
                            st=current_st,
                            reference_st=st[reference],
                            ema5=ema5[index],
                            reference_ema5=ema5[reference],
                            ema8=ema8[index],
                            reference_ema8=ema8[reference],
                            ema13=ema13[index],
                            reference_ema13=ema13[reference],
                            ema26=ema26[index],
                            reference_ema26=ema26[reference],
                        )
                    )

        if lows[index] <= min(lows[index - lookback + 1 : index + 1]):
            offset = previous_extreme_offset(lows, index, lookback - 1, False)
            if offset is None:
                continue
            reference = index - offset
            stars = None
            label = None
            if diff[reference] < diff[index]:
                if st[reference] >= current_st and current_st < -threshold:
                    stars = 1
                elif st[reference] < current_st and current_st < -threshold:
                    stars = 2
                elif st[reference] < current_st and current_st >= -threshold:
                    stars = 3
                label = BULLISH_LABELS.get(stars or 0)
            elif previous_st < 0 and current_st >= 0:
                stars = 4
                label = REVERSAL_UP_LABEL
            if stars is not None and label is not None:
                markers.append(
                    DivergenceStar(
                        timestamp=candles[index].timestamp,
                        side="bullish",
                        stars=stars,
                        label=label,
                        price=lows[index],
                        reference_timestamp=candles[reference].timestamp,
                        reference_price=lows[reference],
                        diff=diff[index],
                        reference_diff=diff[reference],
                        dea=dea[index],
                        reference_dea=dea[reference],
                        st=current_st,
                        reference_st=st[reference],
                        ema5=ema5[index],
                        reference_ema5=ema5[reference],
                        ema8=ema8[index],
                        reference_ema8=ema8[reference],
                        ema13=ema13[index],
                        reference_ema13=ema13[reference],
                        ema26=ema26[index],
                        reference_ema26=ema26[reference],
                    )
                )

    markers.sort(key=lambda marker: (marker.timestamp, marker.side))
    return markers
