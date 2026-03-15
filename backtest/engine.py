"""
Backtesting Engine
Simulates trades on historical price data based on strategy signals.
Computes equity curve and performance metrics.
"""

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_pct: float = 1.0,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 4.0,
    commission: float = 0.001,
) -> dict:
    """
    Parameters
    ----------
    df               : DataFrame with 'Close' and 'signal' columns (1=BUY, -1=SELL, 0=HOLD)
    initial_capital  : Starting capital in USD
    risk_pct         : % of capital to risk per trade
    stop_loss_pct    : Stop-loss distance from entry in %
    take_profit_pct  : Take-profit distance from entry in %
    commission       : Commission per trade as fraction (e.g. 0.001 = 0.1%)

    Returns
    -------
    dict with keys: trades, equity_curve, metrics
    """
    df = df.dropna(subset=["Close", "signal"]).copy()
    df.index = pd.to_datetime(df.index)

    capital = initial_capital
    equity_curve = []
    trades = []

    in_trade = False
    entry_price = 0.0
    entry_date = None
    position_size = 0.0
    sl_price = 0.0
    tp_price = 0.0

    for i, (date, row) in enumerate(df.iterrows()):
        close = float(row["Close"])
        signal = int(row.get("signal", 0))

        # ── Check SL / TP if in trade ──────────────────────────────────────
        if in_trade:
            pnl = 0.0
            exit_reason = None
            exit_price = close

            if close <= sl_price:
                exit_price = sl_price
                exit_reason = "Stop Loss"
            elif close >= tp_price:
                exit_price = tp_price
                exit_reason = "Take Profit"
            elif signal == -1:
                exit_reason = "Signal SELL"

            if exit_reason:
                gross_pnl = (exit_price - entry_price) * position_size
                fee = entry_price * position_size * commission + exit_price * position_size * commission
                pnl = gross_pnl - fee
                capital += pnl
                trades.append(
                    {
                        "entry_date": entry_date.strftime("%Y-%m-%d"),
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "position_size": round(position_size, 4),
                        "pnl": round(pnl, 2),
                        "return_pct": round((exit_price / entry_price - 1) * 100, 3),
                        "exit_reason": exit_reason,
                    }
                )
                in_trade = False

        # ── Open new trade ─────────────────────────────────────────────────
        if not in_trade and signal == 1 and close > 0:
            risk_amount = capital * (risk_pct / 100)
            sl_price = close * (1 - stop_loss_pct / 100)
            tp_price = close * (1 + take_profit_pct / 100)
            risk_per_share = close - sl_price
            if risk_per_share > 0:
                position_size = risk_amount / risk_per_share
                entry_price = close
                entry_date = date
                in_trade = True

        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(capital, 2)})

    # ── Close open trade at end ────────────────────────────────────────────
    if in_trade:
        last_close = float(df["Close"].iloc[-1])
        gross_pnl = (last_close - entry_price) * position_size
        fee = entry_price * position_size * commission + last_close * position_size * commission
        pnl = gross_pnl - fee
        capital += pnl
        trades.append(
            {
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": df.index[-1].strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 4),
                "exit_price": round(last_close, 4),
                "position_size": round(position_size, 4),
                "pnl": round(pnl, 2),
                "return_pct": round((last_close / entry_price - 1) * 100, 3),
                "exit_reason": "End of Data",
            }
        )
        equity_curve[-1]["equity"] = round(capital, 2)

    metrics = _compute_metrics(trades, equity_curve, initial_capital)
    return {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}


def _compute_metrics(trades: list, equity_curve: list, initial_capital: float) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "final_capital": round(initial_capital, 2),
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) * 100
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equities = [e["equity"] for e in equity_curve]
    eq_arr = np.array(equities)
    peak = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peak) / peak * 100
    max_drawdown = float(abs(drawdowns.min()))

    # Daily returns for Sharpe (annualised, risk-free = 0)
    eq_series = pd.Series(eq_arr)
    daily_ret = eq_series.pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    final_capital = equities[-1] if equities else initial_capital
    total_return_pct = (final_capital / initial_capital - 1) * 100

    return {
        "total_trades": len(pnls),
        "win_trades": len(wins),
        "loss_trades": len(losses),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(sum(pnls), 2),
        "total_return_pct": round(total_return_pct, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else 9999,
        "max_drawdown_pct": round(max_drawdown, 3),
        "sharpe_ratio": round(sharpe, 3),
        "final_capital": round(final_capital, 2),
    }
