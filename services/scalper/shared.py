"""Shared utilities for paper scalper engines."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Mapping, MutableMapping, Sequence

TICK_SIZE = 0.01


def normalize_status(value: object, default: str = "inactive") -> str:
    """Return a canonical lower-case status value."""

    if value is None:
        return default
    text = str(value).strip().lower()
    return text or default


def is_active_status(value: object) -> bool:
    """Return ``True`` when ``value`` represents an active status."""

    return normalize_status(value) == "active"


@dataclass(slots=True)
class FeeModel:
    """Simple fee model with per-contract and per-order charges."""

    per_contract: float = 0.65
    per_order: float = 0.0

    def order_fees(self, qty: int) -> float:
        contracts = max(0, int(qty))
        return round(contracts * float(self.per_contract) + float(self.per_order), 2)


def apply_slippage(mid_price: float, *, side: str, ticks: int = 1) -> float:
    """Return a price adjusted by a number of ticks for a buy or sell."""

    ticks = max(0, int(ticks))
    mid = max(0.0, float(mid_price))
    delta = ticks * TICK_SIZE
    if side.lower() == "buy":
        return round(max(TICK_SIZE, mid + delta), 2)
    return round(max(TICK_SIZE, mid - delta), 2)


def calculate_position_size(balance: float, pct_per_trade: float, mid_price: float) -> int:
    pct = max(0.0, float(pct_per_trade)) / 100.0
    mid = max(0.0, float(mid_price))
    if pct <= 0 or mid <= 0:
        return 0
    notional = float(balance) * pct
    cost_per_contract = mid * 100.0
    qty = int(notional // cost_per_contract)
    return max(0, qty)


def summarize_backtest(trades: Sequence[Mapping[str, float]], *, starting_balance: float) -> Mapping[str, float]:
    if not trades:
        return {
            "starting_balance": starting_balance,
            "ending_balance": starting_balance,
            "net_profit": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "losses": 0,
        }
    ending_balance = trades[-1].get("balance", starting_balance)
    total_trades = len(trades)
    wins = sum(1 for trade in trades if float(trade.get("net", 0.0)) > 0)
    losses = sum(1 for trade in trades if float(trade.get("net", 0.0)) < 0)
    net_profit = float(ending_balance) - float(starting_balance)
    win_rate = 0.0 if total_trades == 0 else round(wins / total_trades * 100.0, 2)
    return {
        "starting_balance": float(starting_balance),
        "ending_balance": float(ending_balance),
        "net_profit": net_profit,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "losses": losses,
    }


def compute_trade_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    starting_balance: float,
) -> MutableMapping[str, float]:
    trades = list(rows)
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trades_per_day": 0.0,
        }

    wins: list[float] = []
    losses: list[float] = []
    returns: list[float] = []
    equity: list[float] = [float(starting_balance)]
    trade_dates: set[str] = set()

    balance = float(starting_balance)
    for trade in sorted(trades, key=lambda row: str(row.get("exit_time") or row.get("entry_time"))):
        net = float(trade.get("net_pl") or trade.get("realized_pl") or 0.0)
        roi_pct = trade.get("roi_pct")
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")
        if isinstance(entry_time, str) and entry_time:
            trade_dates.add(entry_time[:10])
        if isinstance(exit_time, str) and exit_time:
            trade_dates.add(exit_time[:10])
        balance += net
        equity.append(balance)
        if net > 0:
            wins.append(net)
        elif net < 0:
            losses.append(net)
        if roi_pct is not None:
            try:
                returns.append(float(roi_pct) / 100.0)
            except Exception:
                pass
        else:
            if starting_balance > 0:
                returns.append(net / float(starting_balance))

    win_rate = len(wins) / total_trades * 100.0
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    profit = sum(wins)
    loss = abs(sum(losses))
    profit_factor = 0.0 if loss == 0 else profit / loss
    sharpe = _simple_sharpe_ratio(returns)
    max_drawdown = _max_drawdown(equity)
    trades_per_day = total_trades / max(1, len(trade_dates))
    return {
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 2),
        "trades_per_day": round(trades_per_day, 2),
    }


def _simple_sharpe_ratio(returns: Sequence[float]) -> float:
    clean = [float(r) for r in returns if isinstance(r, (int, float))]
    if not clean:
        return 0.0
    avg = mean(clean)
    if len(clean) == 1:
        return avg / 1e-9
    stdev = pstdev(clean)
    if stdev == 0:
        return 0.0
    return avg / stdev * (len(clean) ** 0.5)


def _max_drawdown(equity: Sequence[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for value in equity:
        val = float(value)
        peak = max(peak, val)
        if peak <= 0:
            continue
        dd = (val - peak) / peak * 100.0
        max_dd = min(max_dd, dd)
    return max_dd
