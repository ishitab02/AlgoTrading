from __future__ import annotations

import datetime as _dt
import io
import logging
from typing import Dict, Iterable, Optional
import pandas as pd

try:
    import yfinance as yf  
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

import requests
import time

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

def _download_via_yfinance(symbol: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    # Downloads historical data using yfinance.
    if not _HAS_YFINANCE:
        raise ImportError(
            "yfinance is not installed. Install it via `pip install yfinance` to use this function"
        )
    logger.debug("Downloading %s data from yfinance from %s to %s", symbol, start_date, end_date)
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=True)
            if not data.empty:
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            if data.empty:
                logger.warning("No data returned for %s using yfinance", symbol)
            return data
        except Exception as e:
            logger.warning("yfinance download failed for %s (attempt %d/%d): %s", symbol, attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise
    return pd.DataFrame() 

# If yfinance fails or is not available
def _download_via_csv(symbol: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    #Fallback method to download data using Yahoo Finance's CSV endpoint
    logger.debug(
        "Downloading %s data using CSV endpoint from %s to %s (interval %s)",
        symbol,
        start_date,
        end_date,
        interval,
    )
    for attempt in range(MAX_RETRIES):
        try:
            # Convert dates into Unix timestamps
            start_ts = int(_dt.datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int(_dt.datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            url = (
                f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
                f"?period1={start_ts}&period2={end_ts}&interval={interval}&events=history&includeAdjustedClose=true"
            )
            response = requests.get(url)
            response.raise_for_status()
            content = response.text

            # Parse CSV to DataFrame
            data = pd.read_csv(io.StringIO(content))
            if not data.empty and 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
            return data
        except requests.exceptions.RequestException as e:
            logger.warning("CSV download failed for %s (attempt %d/%d): %s", symbol, attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise
    return pd.DataFrame() 

def fetch_stock_data(
    symbols: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = '1d',
    prefer_yfinance: bool = True,
) -> Dict[str, pd.DataFrame]:
    #Fetch historical market data for a list of symbols

    if end_date is None:
        end_date = _dt.date.today().isoformat()
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            if prefer_yfinance and _HAS_YFINANCE:
                df = _download_via_yfinance(sym, start_date, end_date, interval=interval)
            else:
                df = _download_via_csv(sym, start_date, end_date, interval=interval)
        except Exception as e:
            # Attempt fallback if yfinance fails
            if prefer_yfinance and _HAS_YFINANCE:
                logger.warning(
                    "yfinance download failed for %s (%s). Falling back to CSV endpoint.", sym, e
                )
                try:
                    df = _download_via_csv(sym, start_date, end_date, interval=interval)
                except Exception as csv_e:
                    logger.error("CSV fallback also failed for %s: %s", sym, csv_e)
                    df = pd.DataFrame() 
            else:
                logger.error("Data download failed for %s: %s", sym, e)
                df = pd.DataFrame() 
        results[sym] = df
    return results