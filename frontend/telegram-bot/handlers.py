import os

import telegram
import telegram.ext
from api import generate_response, transcribe_audio
from telegram import Update
from tts import SberSpeechAPI

tts_api = SberSpeechAPI(os.getenv("SALUTE_SPEECH_AUTH_TOKEN"))


async def handle_message(update: Update, context: telegram.ext.CallbackContext):
    user_input = update.message.text
    await update.message.chat.send_chat_action("typing")
    response = await generate_response(user_input)
    await update.message.reply_text(response, parse_mode="HTML")


async def handle_voice_message(update: Update, context: telegram.ext.CallbackContext):
    await update.message.chat.send_chat_action("typing")
    audio_file = await update.message.voice.get_file()
    buf = bytearray()
    await audio_file.download_as_bytearray(buf)
    user_input = await transcribe_audio(buf)
    await update.message.chat.send_chat_action("typing")
    response = await generate_response(user_input)
    voice_message = await tts_api.synthesize_text(response, format="opus")
    await update.message.reply_voice(voice_message, caption=response, parse_mode="HTML")
