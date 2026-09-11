"""Polymarket BTC 5m Bollinger mean-reversion strategy core, version 0.1.

This module deliberately contains no wallet or live-order integration.  It is
safe to use in a backtest or paper-trading process first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor, sqrt
from statistics import fmean, pstdev
from typing import Optional, Sequence


class Side(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


@dataclass(frozen=True)
class Candle:
    """A completed five-minute BTC candle."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class OutcomeQuote:
    """Executable quote for the outcome selected by the signal."""

    timestamp: int
    seconds_from_window_open: float
    bid: float
    ask: float
    ask_depth_shares: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class StrategyConfig:
    bankroll: float = 200.0
    bb_period: int = 20
    bb_std_multiplier: float = 2.0
    max_total_cost_per_trade: float = 5.0
    max_entry_seconds: float = 15.0
    min_entry_price: float = 0.35
    max_entry_price: float = 0.65
    max_spread: float = 0.02
    min_ask_depth_multiple: float = 5.0
    min_order_shares: float = 5.0
    min_edge_after_entry_fee: float = 0.02
    taker_fee_rate: float = 0.07
    stop_loss_fraction: float = 0.20
    daily_loss_limit_fraction: float = 0.03
    max_consecutive_losses: int = 3
    entry_is_taker: bool = True
    stop_exit_is_taker: bool = True


@dataclass(frozen=True)
class EntryDecision:
    side: Side
    shares: float
    entry_price: float
    entry_fee: float
    total_cost: float
    stop_bid: float
    estimated_win_probability: float


@dataclass
class RiskState:
    """Daily controls. Reset this state at the start of each trading day."""

    realized_pnl: float = 0.0
    consecutive_losses: int = 0
    has_open_position: bool = False

    def record_closed_trade(self, pnl: float) -> None:
        self.realized_pnl += pnl
        self.has_open_position = False
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0


def fee_per_share(price: float, fee_rate: float = 0.07) -> float:
    """Polymarket crypto taker fee curve, before protocol rounding."""

    if not 0.0 <= price <= 1.0:
        raise ValueError("price must be between 0 and 1")
    if fee_rate < 0:
        raise ValueError("fee_rate cannot be negative")
    return fee_rate * price * (1.0 - price)


def bollinger_bands(
    closes: Sequence[float], period: int = 20, multiplier: float = 2.0
) -> tuple[float, float, float]:
    """Return lower, middle and upper bands using population std deviation."""

    if period < 2:
        raise ValueError("period must be at least 2")
    if len(closes) < period:
        raise ValueError(f"need at least {period} closes")
    window = list(closes[-period:])
    middle = fmean(window)
    deviation = pstdev(window)
    return (
        middle - multiplier * deviation,
        middle,
        middle + multiplier * deviation,
    )


def detect_signal(
    candles: Sequence[Candle], config: StrategyConfig = StrategyConfig()
) -> Optional[Side]:
    """Detect the first close outside a Bollinger band.

    The most recent candle is the signal candle.  A trade, if any, belongs to
    the *next* five-minute Polymarket window.
    """

    if len(candles) < config.bb_period + 1:
        return None

    closes = [candle.close for candle in candles]
    current_lower, _, current_upper = bollinger_bands(
        closes, config.bb_period, config.bb_std_multiplier
    )
    previous_lower, _, previous_upper = bollinger_bands(
        closes[:-1], config.bb_period, config.bb_std_multiplier
    )
    current_close = closes[-1]
    previous_close = closes[-2]

    first_close_below = (
        current_close < current_lower and previous_close >= previous_lower
    )
    if first_close_below:
        return Side.UP

    first_close_above = (
        current_close > current_upper and previous_close <= previous_upper
    )
    if first_close_above:
        return Side.DOWN

    return None


def stop_bid_for_all_in_loss(
    entry_price: float,
    stop_loss_fraction: float = 0.20,
    fee_rate: float = 0.07,
    entry_is_taker: bool = True,
    exit_is_taker: bool = True,
) -> float:
    """Calculate the bid that realizes the requested all-in loss.

    This is a trigger level, not a guaranteed fill price.
    """

    if not 0.0 < entry_price < 1.0:
        raise ValueError("entry_price must be between 0 and 1")
    if not 0.0 < stop_loss_fraction < 1.0:
        raise ValueError("stop_loss_fraction must be between 0 and 1")

    entry_fee = fee_per_share(entry_price, fee_rate) if entry_is_taker else 0.0
    target_net_proceeds = (1.0 - stop_loss_fraction) * (
        entry_price + entry_fee
    )

    if not exit_is_taker or fee_rate == 0.0:
        return target_net_proceeds

    # Solve fee_rate*b^2 + (1-fee_rate)*b - target = 0.
    linear = 1.0 - fee_rate
    discriminant = linear * linear + 4.0 * fee_rate * target_net_proceeds
    bid = (-linear + sqrt(discriminant)) / (2.0 * fee_rate)
    return min(max(bid, 0.0), 1.0)


class BollingerMeanReversionStrategy:
    def __init__(self, config: StrategyConfig = StrategyConfig()) -> None:
        self.config = config

    def daily_loss_limit(self) -> float:
        return self.config.bankroll * self.config.daily_loss_limit_fraction

    def risk_allows_entry(
        self, risk: RiskState, worst_case_loss: float = 0.0
    ) -> bool:
        if worst_case_loss < 0:
            raise ValueError("worst_case_loss cannot be negative")
        if risk.has_open_position:
            return False
        if risk.realized_pnl <= -self.daily_loss_limit():
            return False
        if (
            risk.realized_pnl - worst_case_loss
            < -self.daily_loss_limit()
        ):
            return False
        if risk.consecutive_losses >= self.config.max_consecutive_losses:
            return False
        return True

    def decide_entry(
        self,
        candles: Sequence[Candle],
        quote: OutcomeQuote,
        estimated_win_probability: float,
        risk: RiskState,
    ) -> Optional[EntryDecision]:
        side = detect_signal(candles, self.config)
        if side is None or not self.risk_allows_entry(risk):
            return None
        if not 0.0 <= estimated_win_probability <= 1.0:
            raise ValueError("estimated_win_probability must be between 0 and 1")
        if quote.seconds_from_window_open < 0:
            return None
        if quote.seconds_from_window_open > self.config.max_entry_seconds:
            return None
        if not self.config.min_entry_price <= quote.ask <= self.config.max_entry_price:
            return None
        if quote.bid < 0 or quote.ask > 1 or quote.bid > quote.ask:
            return None
        if quote.spread > self.config.max_spread:
            return None

        entry_fee_each = (
            fee_per_share(quote.ask, self.config.taker_fee_rate)
            if self.config.entry_is_taker
            else 0.0
        )
        required_probability = (
            quote.ask
            + entry_fee_each
            + self.config.min_edge_after_entry_fee
        )
        if estimated_win_probability < required_probability:
            return None

        all_in_cost_each = quote.ask + entry_fee_each
        raw_shares = self.config.max_total_cost_per_trade / all_in_cost_each
        shares = floor(raw_shares * 100.0) / 100.0
        if shares < self.config.min_order_shares:
            return None
        if quote.ask_depth_shares < shares * self.config.min_ask_depth_multiple:
            return None

        entry_fee = entry_fee_each * shares
        total_cost = quote.ask * shares + entry_fee
        if not self.risk_allows_entry(risk, worst_case_loss=total_cost):
            return None
        stop_bid = stop_bid_for_all_in_loss(
            quote.ask,
            self.config.stop_loss_fraction,
            self.config.taker_fee_rate,
            self.config.entry_is_taker,
            self.config.stop_exit_is_taker,
        )
        return EntryDecision(
            side=side,
            shares=shares,
            entry_price=quote.ask,
            entry_fee=entry_fee,
            total_cost=total_cost,
            stop_bid=stop_bid,
            estimated_win_probability=estimated_win_probability,
        )

    def should_stop(self, position: EntryDecision, executable_bid: float) -> bool:
        if not 0.0 <= executable_bid <= 1.0:
            raise ValueError("executable_bid must be between 0 and 1")
        return executable_bid <= position.stop_bid

    def stop_pnl(self, position: EntryDecision, exit_bid: float) -> float:
        exit_fee_each = (
            fee_per_share(exit_bid, self.config.taker_fee_rate)
            if self.config.stop_exit_is_taker
            else 0.0
        )
        net_proceeds = position.shares * (exit_bid - exit_fee_each)
        return net_proceeds - position.total_cost

    @staticmethod
    def settlement_pnl(position: EntryDecision, won: bool) -> float:
        payout = position.shares if won else 0.0
        return payout - position.total_cost


def _demo() -> None:
    config = StrategyConfig()
    strategy = BollingerMeanReversionStrategy(config)
    stop = stop_bid_for_all_in_loss(0.50)
    print("BTC 5m Bollinger mean-reversion strategy")
    print(f"Bankroll: ${config.bankroll:.2f}")
    print(f"Maximum all-in cost per trade: ${config.max_total_cost_per_trade:.2f}")
    print(f"Daily loss limit: ${strategy.daily_loss_limit():.2f}")
    print(f"All-in 20% stop trigger for a 0.50 taker entry: {stop:.4f}")


if __name__ == "__main__":
    _demo()
