"""Historical proxy backtest for the BTC 5m Bollinger strategy.

Coinbase BTC-USD five-minute candles are used as a public proxy for the
Chainlink settlement feed. This backtest evaluates hold-to-resolution
returns. It deliberately does not pretend that BTC OHLC data can reproduce
the path of a Polymarket outcome token, so the 20% token stop is not simulated.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import floor, sqrt
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from btc5m_strategy import (
    Candle,
    RiskState,
    Side,
    StrategyConfig,
    bollinger_bands,
    detect_signal,
    fee_per_share,
)


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


def fetch_coinbase_candles(
    start_timestamp: int,
    end_timestamp: int,
    request_pause_seconds: float = 0.15,
    granularity_seconds: int = GRANULARITY_SECONDS,
) -> list[Candle]:
    """Download completed BTC-USD candles from Coinbase Exchange."""

    if end_timestamp <= start_timestamp:
        raise ValueError("end_timestamp must be after start_timestamp")
    if granularity_seconds not in VALID_GRANULARITIES:
        raise ValueError("unsupported Coinbase candle granularity")

    candles_by_time: dict[int, Candle] = {}
    cursor = start_timestamp
    chunk_seconds = granularity_seconds * 290

    while cursor < end_timestamp:
        chunk_end = min(cursor + chunk_seconds, end_timestamp)
        query = urlencode(
            {
                "granularity": granularity_seconds,
                "start": _rfc3339(cursor),
                "end": _rfc3339(chunk_end),
            }
        )
        request = Request(
            f"{COINBASE_CANDLES_URL}?{query}",
            headers={
                "Accept": "application/json",
                "User-Agent": "polymarket-btc-5m-strategy/0.2",
            },
        )

        last_error: Optional[Exception] = None
        payload = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=20) as response:
                    payload = json.load(response)
                break
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        if payload is None:
            raise RuntimeError(
                f"Coinbase candle request failed near {_rfc3339(cursor)}"
            ) from last_error
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Coinbase response: {payload!r}")

        # Coinbase format: [time, low, high, open, close, volume]
        for row in payload:
            if not isinstance(row, list) or len(row) < 6:
                continue
            timestamp = int(row[0])
            if not start_timestamp <= timestamp < end_timestamp:
                continue
            candles_by_time[timestamp] = Candle(
                timestamp=timestamp,
                low=float(row[1]),
                high=float(row[2]),
                open=float(row[3]),
                close=float(row[4]),
            )

        cursor = chunk_end
        if cursor < end_timestamp:
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


def break_even_win_rate(
    entry_price: float, fee_rate: float = 0.07, entry_is_taker: bool = True
) -> float:
    """Break-even probability when the outcome token is held to settlement."""

    entry_fee = fee_per_share(entry_price, fee_rate) if entry_is_taker else 0.0
    return entry_price + entry_fee


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    probability = wins / total
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    margin = z * sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def run_backtest(
    candles: Sequence[Candle],
    entry_price: float = 0.50,
    config: StrategyConfig = StrategyConfig(),
    enforce_daily_risk_limits: bool = True,
    signal_mode: str = "outside",
) -> BacktestResult:
    """Backtest a Bollinger signal on the next five-minute bar.

    outside trades only the first close beyond an outer band.
    always trades every window after warm-up: below middle is UP, otherwise
    DOWN.
    """

    if len(candles) < config.bb_period + 2:
        raise ValueError("not enough candles for a backtest")
    if not 0.0 < entry_price < 1.0:
        raise ValueError("entry_price must be between 0 and 1")
    if signal_mode not in {"outside", "always"}:
        raise ValueError("signal_mode must be outside or always")

    entry_fee_each = (
        fee_per_share(entry_price, config.taker_fee_rate)
        if config.entry_is_taker
        else 0.0
    )
    all_in_each = entry_price + entry_fee_each
    shares = floor(config.max_total_cost_per_trade / all_in_each * 100.0) / 100.0
    if shares < config.min_order_shares:
        raise ValueError("configured trade budget is below the minimum share count")

    total_cost = shares * all_in_each
    entry_fee = shares * entry_fee_each
    equity = config.bankroll
    peak_equity = equity
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    total_fees = 0.0
    skipped_by_risk = 0
    trades: list[BacktestTrade] = []
    risk = RiskState()
    current_day = None

    first_signal_index = (
        config.bb_period if signal_mode == "outside" else config.bb_period - 1
    )
    for signal_index in range(first_signal_index, len(candles) - 1):
        if signal_mode == "outside":
            signal_window = candles[
                signal_index - config.bb_period : signal_index + 1
            ]
            side = detect_signal(signal_window, config)
        else:
            signal_window = candles[
                signal_index - config.bb_period + 1 : signal_index + 1
            ]
            _, middle, _ = bollinger_bands(
                [candle.close for candle in signal_window],
                config.bb_period,
                config.bb_std_multiplier,
            )
            side = (
                Side.UP
                if signal_window[-1].close <= middle
                else Side.DOWN
            )
        if side is None:
            continue

        signal_candle = candles[signal_index]
        market_candle = candles[signal_index + 1]
        market_day = datetime.fromtimestamp(
            market_candle.timestamp, timezone.utc
        ).date()
        if market_day != current_day:
            risk = RiskState()
            current_day = market_day

        if enforce_daily_risk_limits:
            daily_limit = config.bankroll * config.daily_loss_limit_fraction
            if (
                risk.realized_pnl <= -daily_limit
                or risk.realized_pnl - total_cost < -daily_limit
                or risk.consecutive_losses >= config.max_consecutive_losses
            ):
                skipped_by_risk += 1
                continue

        if equity < total_cost:
            skipped_by_risk += 1
            continue

        market_is_up = market_candle.close >= market_candle.open
        won = (side is Side.UP and market_is_up) or (
            side is Side.DOWN and not market_is_up
        )
        payout = shares if won else 0.0
        pnl = payout - total_cost
        equity += pnl
        total_fees += entry_fee
        if pnl >= 0:
            gross_profit += pnl
        else:
            gross_loss += -pnl
        risk.record_closed_trade(pnl)

        peak_equity = max(peak_equity, equity)
        current_drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, current_drawdown)
        max_drawdown_pct = max(
            max_drawdown_pct,
            current_drawdown / peak_equity * 100.0 if peak_equity else 0.0,
        )
        trades.append(
            BacktestTrade(
                signal_timestamp=signal_candle.timestamp,
                market_timestamp=market_candle.timestamp,
                side=side,
                entry_price=entry_price,
                shares=shares,
                entry_fee=entry_fee,
                total_cost=total_cost,
                btc_open=market_candle.open,
                btc_close=market_candle.close,
                won=won,
                pnl=pnl,
                equity_after=equity,
            )
        )

    wins = sum(trade.won for trade in trades)
    losses = len(trades) - wins
    up_trades = [trade for trade in trades if trade.side is Side.UP]
    down_trades = [trade for trade in trades if trade.side is Side.DOWN]
    ci_low, ci_high = wilson_interval(wins, len(trades))
    total_pnl = equity - config.bankroll
    return BacktestResult(
        signal_mode=signal_mode,
        entry_price=entry_price,
        initial_bankroll=config.bankroll,
        final_equity=equity,
        total_pnl=total_pnl,
        return_pct=total_pnl / config.bankroll * 100.0,
        trade_count=len(trades),
        wins=wins,
        losses=losses,
        win_rate_pct=wins / len(trades) * 100.0 if trades else 0.0,
        break_even_win_rate_pct=break_even_win_rate(
            entry_price, config.taker_fee_rate, config.entry_is_taker
        )
        * 100.0,
        win_rate_ci_low_pct=ci_low * 100.0,
        win_rate_ci_high_pct=ci_high * 100.0,
        total_entry_fees=total_fees,
        average_pnl=total_pnl / len(trades) if trades else 0.0,
        profit_factor=gross_profit / gross_loss if gross_loss else float("inf"),
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        up_trades=len(up_trades),
        up_win_rate_pct=(
            sum(trade.won for trade in up_trades) / len(up_trades) * 100.0
            if up_trades
            else 0.0
        ),
        down_trades=len(down_trades),
        down_win_rate_pct=(
            sum(trade.won for trade in down_trades) / len(down_trades) * 100.0
            if down_trades
            else 0.0
        ),
        skipped_by_risk=skipped_by_risk,
        start_timestamp=candles[0].timestamp,
        end_timestamp=candles[-1].timestamp,
        trades=tuple(trades),
    )


def _format_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def print_results(candles: Sequence[Candle], results: Sequence[BacktestResult]) -> None:
    print("BTC 5m Bollinger mean-reversion proxy backtest")
    print("Source: Coinbase BTC-USD 5m candles (Chainlink proxy)")
    print(
        f"Period: {_format_time(candles[0].timestamp)} -> "
        f"{_format_time(candles[-1].timestamp)}"
    )
    print(f"Candles: {len(candles):,}")
    print(f"Signal mode: {results[0].signal_mode}")
    print("Stop: not simulated; BTC candles cannot reproduce token bid paths")
    print()
    print(
        "Entry  Trades  Win rate  Break-even  P&L      Return   "
        "Max DD   Final"
    )
    for result in results:
        print(
            f"{result.entry_price:>5.2f}"
            f"{result.trade_count:>8}"
            f"{result.win_rate_pct:>9.2f}%"
            f"{result.break_even_win_rate_pct:>11.2f}%"
            f"  {result.total_pnl:>+7.2f}"
            f"  {result.return_pct:>+6.2f}%"
            f"  {result.max_drawdown:>6.2f}"
            f"  {result.final_equity:>7.2f}"
        )
    print()
    primary = results[0]
    print(
        f"95% win-rate interval: {primary.win_rate_ci_low_pct:.2f}% - "
        f"{primary.win_rate_ci_high_pct:.2f}%"
    )
    print(
        f"UP: {primary.up_trades} trades / {primary.up_win_rate_pct:.2f}% wins; "
        f"DOWN: {primary.down_trades} trades / "
        f"{primary.down_win_rate_pct:.2f}% wins"
    )
    print(
        f"Risk-limit skips: {primary.skipped_by_risk}; "
        f"entry fees: USD {primary.total_entry_fees:.2f}"
    )


def save_report(
    path: Path, candles: Sequence[Candle], results: Sequence[BacktestResult]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "methodology": {
            "data_source": "Coinbase Exchange BTC-USD 5m candles",
            "settlement_proxy": "next candle close >= open is UP",
            "token_stop_simulated": False,
            "reason": "BTC OHLC cannot reproduce Polymarket token bid paths",
            "candle_count": len(candles),
            "period_start_utc": _format_time(candles[0].timestamp),
            "period_end_utc": _format_time(candles[-1].timestamp),
        },
        "results": [
            {
                **{
                    key: value
                    for key, value in asdict(result).items()
                    if key != "trades"
                },
                "trades": [
                    {**asdict(trade), "side": trade.side.value}
                    for trade in result.trades
                ],
            }
            for result in results
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def _parse_prices(raw: str) -> list[float]:
    prices = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not prices:
        raise argparse.ArgumentTypeError("at least one entry price is required")
    if any(not 0.0 < price < 1.0 for price in prices):
        raise argparse.ArgumentTypeError("entry prices must be between 0 and 1")
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the BTC 5m Bollinger mean-reversion signal."
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/coinbase_btcusd_5m_30d.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/backtest_30d.json"),
    )
    parser.add_argument(
        "--entry-prices",
        type=_parse_prices,
        default=[0.50, 0.51, 0.52],
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download candles even when the CSV already exists",
    )
    parser.add_argument(
        "--ignore-risk-limits",
        action="store_true",
        help="measure every signal without daily risk stops",
    )
    parser.add_argument(
        "--signal-mode",
        choices=["outside", "always"],
        default="outside",
        help="outside trades outer-band excursions; always trades every window",
    )
    args = parser.parse_args()

    if args.days <= 0:
        parser.error("--days must be positive")

    if args.refresh or not args.data.exists():
        now = datetime.now(timezone.utc)
        completed_end = int(now.timestamp()) // GRANULARITY_SECONDS
        completed_end *= GRANULARITY_SECONDS
        start = completed_end - int(timedelta(days=args.days).total_seconds())
        candles = fetch_coinbase_candles(start, completed_end)
        save_candles_csv(args.data, candles)
    else:
        candles = load_candles_csv(args.data)

    if not candles:
        raise RuntimeError("no candles available")
    cutoff = candles[-1].timestamp - int(
        timedelta(days=args.days).total_seconds()
    )
    candles = [candle for candle in candles if candle.timestamp > cutoff]

    config = StrategyConfig()
    results = [
        run_backtest(
            candles,
            entry_price=price,
            config=config,
            enforce_daily_risk_limits=not args.ignore_risk_limits,
            signal_mode=args.signal_mode,
        )
        for price in args.entry_prices
    ]
    print_results(candles, results)
    save_report(args.report, candles, results)
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
