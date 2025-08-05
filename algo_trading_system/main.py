from __future__ import annotations

import logging
import os
from typing import List
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .data_fetch import fetch_stock_data
from .strategy import generate_signals, backtest_signals
from .ml_model import prepare_features, train_models
from .telegram_notifier import send_telegram_message

try:
    from .google_sheets import GoogleSheetsLogger
    _HAS_SHEETS = True
except ImportError:
    _HAS_SHEETS = False

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def run_demo(symbols: List[str], start_date: str, end_date: str, log_to_sheets: bool = False) -> None:
    try:
        send_telegram_message(f"Algo-trading system started for symbols: {', '.join(symbols)}")
    except Exception as e:
        logger.warning("Failed to send Telegram start notification: %s", e)

    logger.debug("Fetching data for symbols: %s", symbols)
    data = fetch_stock_data(symbols, start_date, end_date)
    # Google Sheets logging
    sheets_logger = None
    if log_to_sheets and _HAS_SHEETS:
        creds_path = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        sheet_name = os.environ.get('GOOGLE_SHEET_NAME')
        if not creds_path or not sheet_name:
            raise RuntimeError(
                "GOOGLE_CREDENTIALS_JSON and GOOGLE_SHEET_NAME environment variables must be set to log to Sheets."
            )
        sheets_logger = GoogleSheetsLogger(creds_path, sheet_name)
    all_trades = pd.DataFrame()
    summary_rows = []
    # Processing each stock individually
    for sym, df in data.items():
        logger.info("Processing %s", sym)
        # Droping rows with missing values
        df = df.dropna(subset=["Close", "Volume"])
        df_signals = generate_signals(df)
        trades_df, summary = backtest_signals(df_signals)
        summary['symbol'] = sym
        summary_rows.append(summary)
        trades_df['symbol'] = sym
        all_trades = pd.concat([all_trades, trades_df], ignore_index=True)

        # Machine-learning component
        try:
            X, y = prepare_features(df)
            metrics = train_models(X, y)
            summary.update(metrics)
            logger.info(
                "%s: logistic_accuracy=%.3f, tree_accuracy=%.3f",
                sym,
                metrics['logistic_accuracy'],
                metrics['tree_accuracy'],
            )
        except Exception as e:
            logger.warning("ML training skipped for %s due to error: %s", sym, e)
        # Optionally log signals
        if sheets_logger:
            sheets_logger.log_signals(df_signals.reset_index())
    # Summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    logger.debug("Completed processing! Summary:\n%s", summary_df)

    # Telegram notification 
    try:
        total_trades = summary_df['total_trades'].sum()
        total_wins = summary_df['wins'].sum()  

        # Calculation of the average win rate
        if total_trades > 0:
            avg_win_rate = (total_wins / total_trades) * 100
        else:
            avg_win_rate = 0.0

        message = f"Algo-trading system completed!\n\n"
        message += f"Total trades: {int(total_trades)}\n"
        message += f"Average win rate: {avg_win_rate:.1f}%\n"
        message += f"Symbols processed: {len(symbols)}"
        send_telegram_message(message)
    except Exception as e:
        logger.warning("Failed to send Telegram completion notification: %s", e)

    if sheets_logger:
        sheets_logger.log_trades(all_trades)
        sheets_logger.log_summary(summary_df)


if __name__ == '__main__':
    # Dynamic date calculation 
    end_date_obj = datetime.now()
    start_date_obj = end_date_obj - relativedelta(months=6)

    START_DATE = start_date_obj.strftime('%Y-%m-%d')
    END_DATE = end_date_obj.strftime('%Y-%m-%d')

    SYMBOLS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "BHARTIARTL.NS", "SBIN.NS"
    ]

    LOG_TO_SHEETS = bool(os.environ.get('GOOGLE_CREDENTIALS_JSON'))

    # Run with 6 month dynamic dates
    print(f"Execution of 6 month backtest from {START_DATE} to {END_DATE} is as follows:")
    run_demo(SYMBOLS, START_DATE, END_DATE, log_to_sheets=LOG_TO_SHEETS)
