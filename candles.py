"""Coinbase BTC-USD candle download, caching, and loading."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Candle:
    """A completed BTC candle."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float


COINBASE_CANDLES_URL = (
    "https://api.exchange.coinbase.com/products/BTC-USD/candles"
)
GRANULARITY_SECONDS = 300
VALID_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}


@dataclass(frozen=True)
class BacktestTrade:
    signal_timestamp: int
    market_timestamp: int
    side: Side
    entry_price: float
    shares: float
    entry_fee: float
    total_cost: float
    btc_open: float
    btc_close: float
    won: bool
    pnl: float
    equity_after: float


@dataclass(frozen=True)
class BacktestResult:
    signal_mode: str
    entry_price: float
    initial_bankroll: float
    final_equity: float
    total_pnl: float
    return_pct: float
    trade_count: int
    wins: int
    losses: int
    win_rate_pct: float
    break_even_win_rate_pct: float
    win_rate_ci_low_pct: float
    win_rate_ci_high_pct: float
    total_entry_fees: float
    average_pnl: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    up_trades: int
    up_win_rate_pct: float
    down_trades: int
    down_win_rate_pct: float
    skipped_by_risk: int
    start_timestamp: int
    end_timestamp: int
    trades: tuple[BacktestTrade, ...]


def _rfc3339(timestamp: int) -> str:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _coinbase_page(
    start_timestamp: int,
    end_timestamp: int,
    granularity_seconds: int,
) -> dict[int, Candle]:
    """Request one page of candles and index it by open time."""

    query = urlencode(
        {
            "granularity": granularity_seconds,
            "start": _rfc3339(start_timestamp),
            "end": _rfc3339(end_timestamp),
        }
    )
    request = Request(
        f"{COINBASE_CANDLES_URL}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "btc-5m-direction-model/0.1",
        },
    )

    last_error: Optional[Exception] = None
    payload = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.load(response)
            break
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            if attempt < 3:
                # Coinbase answers rapid repeats with 400/429; back off firmly.
                time.sleep(1.5 * (attempt + 1) ** 1.5)
    if payload is None:
        raise RuntimeError(
            f"Coinbase candle request failed near {_rfc3339(start_timestamp)}"
        ) from last_error
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Coinbase response: {payload!r}")

    page: dict[int, Candle] = {}
    # Coinbase format: [time, low, high, open, close, volume]
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        timestamp = int(row[0])
        if not start_timestamp <= timestamp < end_timestamp:
            continue
        page[timestamp] = Candle(
            timestamp=timestamp,
            low=float(row[1]),
            high=float(row[2]),
            open=float(row[3]),
            close=float(row[4]),
        )
    return page


def fetch_coinbase_candles(
    start_timestamp: int,
    end_timestamp: int,
    request_pause_seconds: float = 0.15,
    granularity_seconds: int = GRANULARITY_SECONDS,
) -> list[Candle]:
    """Download BTC-USD candles from Coinbase Exchange.

    Coinbase sometimes returns a page that stops short of the requested end,
    which used to leave a silent gap at the newest end of the range. Each page
    is now capped well below the exchange limit and the tail is re-requested
    until it is covered or stops making progress.
    """

    if end_timestamp <= start_timestamp:
        raise ValueError("end_timestamp must be after start_timestamp")
    if granularity_seconds not in VALID_GRANULARITIES:
        raise ValueError("unsupported Coinbase candle granularity")

    candles_by_time: dict[int, Candle] = {}
    # Stay comfortably under the exchange's 300-row page limit.
    chunk_seconds = granularity_seconds * 150
    cursor = start_timestamp
    while cursor < end_timestamp:
        chunk_end = min(cursor + chunk_seconds, end_timestamp)
        candles_by_time.update(
            _coinbase_page(cursor, chunk_end, granularity_seconds)
        )
        cursor = chunk_end
        if cursor < end_timestamp:
            time.sleep(request_pause_seconds)

    # Repair a short tail: keep asking for the missing end of the range.
    for _ in range(4):
        newest = max(candles_by_time, default=None)
        if newest is None or newest >= end_timestamp - granularity_seconds:
            break
        try:
            tail = _coinbase_page(
                newest + granularity_seconds, end_timestamp, granularity_seconds
            )
        except Exception:
            # A failed repair leaves the gap for the caller to forward-fill.
            break
        if not tail:
            break
        candles_by_time.update(tail)
        if max(tail) <= newest:
            break
        time.sleep(request_pause_seconds)

    return [candles_by_time[key] for key in sorted(candles_by_time)]


def save_candles_csv(path: Path, candles: Iterable[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close"],
        )
        writer.writeheader()
        for candle in candles:
            writer.writerow(asdict(candle))


def load_candles_csv(path: Path) -> list[Candle]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
            for row in csv.DictReader(handle)
        ]


