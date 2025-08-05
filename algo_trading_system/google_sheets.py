from __future__ import annotations

import logging
import pandas as pd

try:
    import gspread  
    from gspread_dataframe import set_with_dataframe  
    from oauth2client.service_account import ServiceAccountCredentials  
    _HAS_GSPREAD = True
except ImportError:
    _HAS_GSPREAD = False

logger = logging.getLogger(__name__)


class GoogleSheetsLogger:
    # Initializes the Google Sheets logger with credentials and spreadsheet name
    def __init__(self, creds_json_path: str, spreadsheet_name: str) -> None:
        if not _HAS_GSPREAD:
            raise ImportError(
                "gspread and oauth2client are required for Google Sheets logging!"
            )
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive',
        ]
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                creds_json_path, scopes=scope
            )
            gc = gspread.authorize(credentials)
            self.sheet = gc.open(spreadsheet_name)
            logger.info("Connected to Google Sheet: %s", spreadsheet_name)
        except gspread.exceptions.SpreadsheetNotFound:
            logger.error("Google Sheet ", spreadsheet_name, " not found. Please ensure the sheet exists and the service account has access.")
            raise
        except Exception as e:
            logger.error("Failed to connect to Google Sheets: %s", e)
            raise
    def write_dataframe(self, worksheet_title: str, df: pd.DataFrame) -> None:
        # Write a DataFrame to a specified worksheet in the Google Sheet
        try:
            try:
                worksheet = self.sheet.worksheet(worksheet_title)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = self.sheet.add_worksheet(title=worksheet_title, rows=1000, cols=20)

            # Clear existing contents
            worksheet.clear()

            # Use gspread_dataframe helper to write DataFrame
            set_with_dataframe(worksheet, df, include_index=False)
            logger.info("Wrote %d rows to worksheet '%s'", len(df), worksheet_title)
        except Exception as exc:
            logger.error("Failed to write DataFrame to Google Sheets: %s", exc)
            raise
    
    # Log trades to the 'Trades' worksheet
    def log_trades(self, trades_df: pd.DataFrame) -> None:
        self.write_dataframe('Trades', trades_df)

    # Log summary metrics to the 'Summary' worksheet
    def log_summary(self, summary: dict) -> None:
        summary_df = summary
        self.write_dataframe('Summary', summary_df)

    # Log signals to the 'Signals' worksheet
    def log_signals(self, signals_df: pd.DataFrame) -> None:
        self.write_dataframe('Signals', signals_df)