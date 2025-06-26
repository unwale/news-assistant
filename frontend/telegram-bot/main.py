import asyncio
import os
import time

from handlers import handle_message, handle_voice_message
from telegram.ext import ApplicationBuilder, MessageHandler, filters

TOKEN = os.getenv("BOT_TOKEN")


def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    while True:
        try:
            app.run_polling(bootstrap_retries=-1, drop_pending_updates=True)
        except:
            time.sleep(1)


if __name__ == "__main__":
    main()
