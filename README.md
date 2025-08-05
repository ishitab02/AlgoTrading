# Algo-Trading System with ML & Automation

This project is a Python-based prototype of an algorithmic trading system. It automatically fetches stock data for NIFTY 50 tickers, applies a "buy the dip" trading strategy, backtests its performance, and uses machine learning to predict future price movements. All results, trades, and analytics are logged to Google Sheets, with real-time notifications sent via Telegram.

## Key Features

- **Automated Data Ingestion:** Fetches daily stock data from Yahoo Finance with built-in resilience (retries and fallbacks).
- **Rule-Based Trading Strategy:** Implements a "buy the dip" strategy using RSI and Moving Average indicators.
- **Performance Backtesting:** Calculates detailed performance metrics, including P&L, win rate, and cumulative returns over a dynamic 6-month period.
- **ML-Powered Predictions:** Utilizes Decision Tree and Logistic Regression models to forecast next-day price direction, validated correctly with `TimeSeriesSplit`.
- **Cloud Integration:** Logs all trades, daily signals, and summary analytics to Google Sheets in separate, organized tabs.
- **Real-Time Alerts:** Sends start, completion, and error notifications to a Telegram chat.

## Project Structure

The project is built with a modular architecture to ensure clean separation of concerns:

```
algo_trading_system/
├── __init__.py
├── main.py              # Main entry point to run the system
├── data_fetch.py        # Module for data ingestion from APIs
├── indicators.py        # Functions for technical indicator calculations
├── strategy.py          # Trading strategy logic and backtesting engine
├── ml_model.py          # Machine learning models and feature preparation
├── google_sheets.py     # Google Sheets integration and logging
└── telegram_notifier.py # Module for sending Telegram notifications
```

## Setup & Execution

#### 1. Prerequisites

- Python 3.8+
- A configured Python virtual environment (e.g., `venv` or `conda`).

#### 2. Installation

1.  Clone this repository
```
git clone https://github.com/your-username/AlgoTrading.git
```
2.  Navigate to the project directory
   ```
cd AlgoTrading
```
4.  Install all required packages:
    ```bash
    pip install -r requirements.txt
    ```
5.  Create a `.env` file in the project's root directory and add your credentials. See `.env.example` for the required format.
    ```
    TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
    TELEGRAM_CHAT_ID="your_telegram_chat_id"
    GOOGLE_CREDENTIALS_JSON="secrets/service_account.json"
    GOOGLE_SHEET_NAME="YourGoogleSheetName"
    ```
6.  Place your Google Cloud `service_account.json` file inside the `/secrets` directory.

#### 3. Execution

To run the entire system, execute the main module from the project's root directory. The script will perform a 6-month backtest ending on the current day.

```bash
python -m algo_trading_system.main
```

## Strategy Deep Dive

The core strategy is to identify a short-term buying opportunity within a confirmed long-term uptrend.

#### Buy Conditions

- **RSI < 30:** The stock is considered significantly oversold.
- **20-day SMA > 50-day SMA:** The stock is in a confirmed bullish trend (a "Golden Cross" state).

#### Sell Conditions

A position is exited under two conditions to manage risk and lock in profits:

- **RSI > 70:** The stock is considered overbought.
- **Death Cross:** The 20-day SMA crosses _below_ the 50-day SMA, signaling a potential trend reversal.

### A Note on the Low Trade Count

The backtest results may show a low number of trades. This is an expected outcome and a feature of this highly selective strategy. The combination of an _extremely oversold_ condition (`RSI < 30`) occurring within an _established strong uptrend_ (`20-DMA > 50-DMA`) is inherently rare. This demonstrates that the system is working correctly by filtering for very specific, high-conviction setups as designed.

## Machine Learning Analytics

The project includes a bonus ML component to forecast next-day price direction.

- **Models:** A Logistic Regression and a Decision Tree classifier.
- **Features:** A rich set of features is used, including RSI, MACD, various SMAs, volume indicators, volatility, and lagged values.
- **Validation:** The models are validated using `TimeSeriesSplit` cross-validation. This is crucial for financial data as it respects the chronological order and prevents lookahead bias, providing a realistic measure of performance. The resulting low accuracy scores (~50%) are an honest reflection of the difficulty in predicting financial markets with basic models.

## Outputs & Deliverables

The system generates the following outputs:

- **Console Logs:** Real-time status updates and trade details.
- **Google Sheets:** Three organized tabs:
  1.  `Trades`: A detailed log of every completed trade.
  2.  `Summary`: High-level performance metrics per stock, including P&L, win rate, and ML accuracy.
  3.  `Signals`: A daily log of all indicator values for debugging and analysis.
- **Telegram Notifications:** Alerts for the start and successful completion of the script, including a final summary.
