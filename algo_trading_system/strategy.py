from __future__ import annotations
import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from .indicators import calculate_rsi, calculate_macd, calculate_sma

logger = logging.getLogger(__name__)

# Generates trading signals based on RSI and moving averages
def generate_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    short_window: int = 20,
    long_window: int = 50,
) -> pd.DataFrame:
    
    df = df.copy()
    df["rsi"] = calculate_rsi(df["Close"], period=rsi_period)
    df["sma_short"] = calculate_sma(df["Close"], short_window)
    df["sma_long"] = calculate_sma(df["Close"], long_window)
    df["ma_diff"] = df["sma_short"] - df["sma_long"]
    df["ma_diff_prev"] = df["ma_diff"].shift(1)
    logger.debug("DataFrame after indicator calculation:\n%s", df.tail())
    df["signal"] = 0

    # Buys when RSI is below 30, confirming a short-term dip, but only while the 20-DMA is already above the 50-DMA, confirming a broader uptrend
    buy_conditions = (
        (df["rsi"] < 30) &      
        (df["ma_diff"] > 0)     
    )

    df.loc[buy_conditions, "signal"] = 1
    return df

# Backtest the trading signals and compute P&L metrics
def backtest_signals(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    rsi_exit: float = 70.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    trades: List[Dict[str, object]] = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    shares = 0.0
    capital = initial_capital

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        price = row["Close"]

        if not in_position and row["signal"] == 1:
            entry_price = price
            entry_date = date
            shares = capital / entry_price
            in_position = True
            logger.debug("Entering position on %s at %.2f", date, price)
            continue

        if in_position:
            rsi_val = row["rsi"]
            ma_diff = row["ma_diff"]
            ma_diff_prev = row["ma_diff_prev"]
            cross_down = ma_diff_prev >= 0 and ma_diff < 0
            overbought = rsi_val > rsi_exit

            if cross_down or overbought or i == len(df) - 1:
                exit_price = price
                exit_date = date
                pnl = (exit_price - entry_price) * shares
                pnl_pct = pnl / (entry_price * shares)
                capital += pnl
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
                logger.debug(
                    "Exiting position on %s at %.2f (pnl %.2f, %.2f%%)",
                    exit_date,
                    exit_price,
                    pnl,
                    pnl_pct * 100,
                )
                in_position = False
                entry_price = 0.0
                shares = 0.0

    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df["pnl_pct"] = trade_df["pnl_pct"] * 100

    # Summary statistics of trades
    summary = {
        "total_trades": len(trade_df),
        "wins": int((trade_df["pnl"] > 0).sum()) if not trade_df.empty else 0,
        "losses": int((trade_df["pnl"] <= 0).sum()) if not trade_df.empty else 0,
        "win_rate": float((trade_df["pnl"] > 0).mean()) if not trade_df.empty else 0.0,
        "cumulative_return_pct": float((capital / initial_capital - 1) * 100),
    }

    return trade_df, summary