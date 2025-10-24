import asyncio
import logging
import os
from io import BytesIO
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
import google.generativeai as genai
from PIL import Image

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Получение переменных окружения ---
TELEGRAM_TOKEN = os.getenv('BOT_TOKEN') # Используем BOT_TOKEN
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Проверка на наличие ключей
if not TELEGRAM_TOKEN:
    logger.error("Переменная окружения 'BOT_TOKEN' не установлена!")
    exit(1)
if not GEMINI_API_KEY:
    logger.error("Переменная окружения 'GEMINI_API_KEY' не установлена!")
    exit(1)

# --- Инициализация ---
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- Настройка Gemini ---
genai.configure(api_key=GEMINI_API_KEY)

# Используем доступную модель Gemini
MODEL_NAME = "gemini-2.5-pro" # или gemini-2.0-flash, или gemini-flash-latest
try:
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info(f"Модель {MODEL_NAME} успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели {MODEL_NAME}: {e}")
    exit(1)

# --- Промпт для анализа еды ---
CALORIE_PROMPT = """Проанализируй это изображение еды и предоставь детальную оценку калорийности.

Ответь в следующем формате:

🍽 **ЧТО НА ФОТО:**
[Перечисли все блюда и продукты, которые видишь]

📊 **КАЛОРИЙНОСТЬ ПО ПОЗИЦИЯМ:**
[Для каждого блюда укажи:
- Название и примерный размер порции
- Калории
- Белки, Жиры, Углеводы]

🔢 **ИТОГО:**
Калории: [общее число] ккал
Белки: [число] г
Жиры: [число] г
Углеводы: [число] г

💡 **КОММЕНТАРИЙ:**
[Краткий комментарий о питательности блюда]

Будь максимально точным в оценках. Если не можешь точно определить блюдо, укажи это."""


@dp.message(CommandStart())
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    await message.answer(
        "👋 Привет! Я бот для подсчёта калорий.\n\n"
        "📸 Просто отправь мне фото еды, и я оценю:\n"
        "• Количество калорий\n"
        "• Белки, жиры, углеводы\n"
        "• Питательную ценность\n\n"
        "Попробуй прямо сейчас!"
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    """Обработчик фотографий"""
    try:
        # Отправляем сообщение о начале обработки
        processing_msg = await message.answer("🔍 Анализирую фото...")

        # Получаем файл
        photo = message.photo[-1]  # Берём фото наибольшего размера
        file = await bot.get_file(photo.file_id)

        # Скачиваем фото (возвращает bytes)
        file_bytes = await bot.download_file(file.file_path)

        # Конвертируем bytes в PIL Image
        image = Image.open(BytesIO(file_bytes))

        # Отправляем на анализ в Gemini
        response = model.generate_content([CALORIE_PROMPT, image])

        # Удаляем сообщение о обработке
        await processing_msg.delete()

        # --- Обработка ответа Gemini ---
        if response.text:
            await message.answer(response.text)
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.warning(f"Gemini blocked the prompt: {response.prompt_feedback.block_reason}")
            await message.answer("❌ Не удалось проанализировать изображение из-за ограничений.")
        else:
            logger.warning("Gemini вернул пустой ответ или только изображение.")
            await message.answer("❌ Не удалось проанализировать изображение. Попробуйте другое фото.")

    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
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
    logger.info("Бот запущен!")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == '__main__':
    asyncio.run(main())
    