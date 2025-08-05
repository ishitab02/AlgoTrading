import os
import logging
import requests

logger = logging.getLogger(__name__)

# This function fetches the bot token and chat ID from environment variables at the time of execution.
def send_telegram_message(message: str) -> None:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram bot token or chat ID not set. Skipping Telegram notification.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()  
        logger.debug("Telegram message sent successfully!")
    except requests.exceptions.RequestException as e:
        logger.error("Failed to send Telegram message: %s", e)

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    send_telegram_message("Test message from telegram_notifier.py direct execution!")