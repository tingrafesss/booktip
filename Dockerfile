FROM python:3.12-slim

# Утилита pdfunite (склейка PDF для "потоковой" импозиции)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Базовые переменные окружения и UTF-8
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# Сначала зависимости — так кэш лучше работает
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Затем код
COPY . /app

# Порт и запуск
ENV PORT=8000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

