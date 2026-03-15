"""
SMA Crossover Strategy
BUY  when fast SMA crosses above slow SMA
SELL when fast SMA crosses below slow SMA
"""

import pandas as pd


def generate_signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """
    Add SMA crossover signals to a price DataFrame.

    Parameters
    ----------
    df   : DataFrame with at least a 'Close' column (DatetimeIndex)
    fast : fast SMA window  (default 50)
    slow : slow SMA window  (default 200)

    Returns
    -------
    df with added columns: SMA_fast, SMA_slow, signal
    signal values: 1 = BUY, -1 = SELL, 0 = HOLD
    """
    df = df.copy()
    df[f"SMA_{fast}"] = df["Close"].rolling(fast).mean()
    df[f"SMA_{slow}"] = df["Close"].rolling(slow).mean()

    df["signal"] = 0

    # Crossover detection
    df["_prev_fast"] = df[f"SMA_{fast}"].shift(1)
    df["_prev_slow"] = df[f"SMA_{slow}"].shift(1)

    # BUY: fast crosses above slow
    buy_mask = (df["_prev_fast"] <= df["_prev_slow"]) & (
        df[f"SMA_{fast}"] > df[f"SMA_{slow}"]
    )
    df.loc[buy_mask, "signal"] = 1

    # SELL: fast crosses below slow
    sell_mask = (df["_prev_fast"] >= df["_prev_slow"]) & (
        df[f"SMA_{fast}"] < df[f"SMA_{slow}"]
    )
    df.loc[sell_mask, "signal"] = -1

    df.drop(columns=["_prev_fast", "_prev_slow"], inplace=True)
    return df
