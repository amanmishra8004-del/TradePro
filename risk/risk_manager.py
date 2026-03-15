"""
Risk Management Module
Calculates position size, stop-loss price, take-profit price.
"""


def calculate_position(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict:
    """
    Parameters
    ----------
    capital        : Total account capital (USD)
    risk_pct       : Risk per trade as % of capital  (e.g. 1.0 for 1%)
    entry_price    : Current price of the asset
    stop_loss_pct  : Stop-loss distance as % from entry (e.g. 2.0 for 2%)
    take_profit_pct: Take-profit distance as % from entry (e.g. 4.0 for 4%)

    Returns
    -------
    dict with: risk_amount, stop_loss_price, take_profit_price,
               position_size (shares), position_value, risk_reward_ratio
    """
    risk_amount = capital * (risk_pct / 100)
    stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
    take_profit_price = entry_price * (1 + take_profit_pct / 100)
    risk_per_share = entry_price - stop_loss_price

    if risk_per_share <= 0:
        position_size = 0.0
    else:
        position_size = risk_amount / risk_per_share

    position_value = position_size * entry_price
    risk_reward = (take_profit_price - entry_price) / (entry_price - stop_loss_price)

    return {
        "capital": round(capital, 2),
        "risk_amount": round(risk_amount, 2),
        "entry_price": round(entry_price, 4),
        "stop_loss_price": round(stop_loss_price, 4),
        "take_profit_price": round(take_profit_price, 4),
        "position_size": round(position_size, 4),
        "position_value": round(position_value, 2),
        "risk_reward_ratio": round(risk_reward, 2),
    }
