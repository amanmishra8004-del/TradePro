"""
RSI Strategy
BUY  when RSI < oversold threshold (default 30)
SELL when RSI > overbought threshold (default 70)
"""

import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def generate_signals(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    """
    Add RSI signals to a price DataFrame.

    Returns
    -------
    df with added columns: RSI, signal
    signal values: 1 = BUY, -1 = SELL, 0 = HOLD
    """
    df = df.copy()
    df["RSI"] = _rsi(df["Close"], period)
    df["signal"] = 0

    prev_rsi = df["RSI"].shift(1)

    # BUY: RSI crosses above oversold
    buy_mask = (prev_rsi <= oversold) & (df["RSI"] > oversold)
    df.loc[buy_mask, "signal"] = 1

    # SELL: RSI crosses above overbought
    sell_mask = (prev_rsi <= overbought) & (df["RSI"] > overbought)
    df.loc[sell_mask, "signal"] = -1

    return df
