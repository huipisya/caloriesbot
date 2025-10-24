import asyncio
import logging
import os
import base64
from io import BytesIO
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import CommandStart
import google.generativeai as genai
from PIL import Image

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Получение переменных окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Инициализация
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Настройка Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Промпт для анализа еды
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
        "Попробуй прямо сейчас!!"
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
        
        # Скачиваем фото
        photo_bytes = await bot.download_file(file.file_path)
        
        # Конвертируем в PIL Image
        image = Image.open(BytesIO(photo_bytes.read()))
        
        # Отправляем на анализ в Gemini
        response = model.generate_content([CALORIE_PROMPT, image])
        
        # Удаляем сообщение о обработке
        await processing_msg.delete()
        
        # Отправляем результат
        if response.text:
            await message.answer(response.text)
        else:
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
        "Я не могу анализировать текст - только изображения блюд."
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