import asyncio
import logging
import os
import base64
from io import BytesIO
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from groq import Groq
from PIL import Image

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Получение переменных окружения ---
TELEGRAM_TOKEN = os.getenv('BOT_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Проверка на наличие ключей
if not TELEGRAM_TOKEN:
    logger.error("Переменная окружения 'BOT_TOKEN' не установлена!")
    exit(1)
if not GROQ_API_KEY:
    logger.error("Переменная окружения 'GROQ_API_KEY' не установлена!")
    exit(1)

# --- Инициализация ----
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- Настройка Groq ---
client = Groq(api_key=GROQ_API_KEY)

# --- Промпт для анализа еды ---
CALORIE_PROMPT = """Проанализируй это изображение еды и предоставь детальную информацию:

📋 **Определи блюда**: Что изображено на фото

📏 **Оценка порций**: Примерный размер порций

📊 **Калорийность и БЖУ**:
• Калории: [число] ккал
• Белки: [число] г
• Жиры: [число] г
• Углеводы: [число] г

💡 **Рекомендации**: Краткая оценка питательной ценности

Будь максимально точным в оценке."""


@dp.message(CommandStart())
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    await message.answer(
        "👋 Привет! Я бот для подсчёта калорий на базе Groq AI (⚡ сверхбыстрый!).\n\n"
        "📸 Просто отправь мне фото еды, и я оценю:\n"
        "• Количество калорий\n"
        "• Белки, жиры, углеводы\n"
        "• Питательную ценность\n\n"
        "Попробуй прямо сейчас! 🚀"
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    """Обработчик фотографий"""
    try:
        # Отправляем сообщение о начале обработки
        processing_msg = await message.answer("⚡ Анализирую фото с помощью Groq AI...")

        # Получаем файл
        photo = message.photo[-1]  # Берём фото наибольшего размера
        file = await bot.get_file(photo.file_id)

        # Скачиваем фото
        file_bytes_io = await bot.download_file(file.file_path)

        # Конвертируем в base64 для Groq API
        image_data = file_bytes_io.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Определяем тип изображения
        image_media_type = "image/jpeg"  # По умолчанию
        try:
            img = Image.open(BytesIO(image_data))
            if img.format == "PNG":
                image_media_type = "image/png"
            elif img.format == "WEBP":
                image_media_type = "image/webp"
            elif img.format == "JPEG" or img.format == "JPG":
                image_media_type = "image/jpeg"
        except Exception as e:
            logger.warning(f"Не удалось определить формат изображения: {e}")

        # Отправляем на анализ в Groq (используем модель с vision)
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # Актуальная модель
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": CALORIE_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_media_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        # Удаляем сообщение о обработке
        await processing_msg.delete()

        # Извлекаем текст ответа
        if completion.choices and len(completion.choices) > 0:
            answer_text = completion.choices[0].message.content
            await message.answer(f"🤖 *Анализ от Groq AI:*\n\n{answer_text}", parse_mode="Markdown")
        else:
            await message.answer("❌ Не удалось получить ответ от Groq AI. Попробуйте ещё раз.")

    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}", exc_info=True)
        
        # Более информативное сообщение об ошибке
        error_message = str(e)
        if "rate_limit" in error_message.lower():
            await message.answer("⏳ Превышен лимит запросов. Подождите немного и попробуйте снова.")
        elif "invalid" in error_message.lower():
            await message.answer(
                "❌ Не удалось обработать изображение.\n"
                "Попробуйте:\n"
                "• Отправить фото в лучшем качестве\n"
                "• Убедиться, что еда хорошо видна"
            )
        else:
            await message.answer(
                "❌ Произошла ошибка при анализе фото.\n"
                "Пожалуйста, попробуйте:\n"
                "• Сделать более чёткое фото\n"
                "• Убедиться, что еда хорошо видна\n"
                "• Отправить фото в хорошем освещении"
            )


@dp.message(F.text)
async def handle_text(message: Message):
    """Обработчик текстовых сообщений"""
    await message.answer(
        "📸 Отправьте мне фото еды, чтобы я мог посчитать калории!\n\n"
        "Я не могу анализировать текст - только изображения."
    )


async def main():
    """Запуск бота"""
    logger.info("Бот на Groq AI запущен! ⚡")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == '__main__':
    asyncio.run(main())
