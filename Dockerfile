FROM python:3.12-slim

# LibreOffice для конвертации DOCX→PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    fonts-dejavu-core \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Локаль (на всякий)
RUN sed -i 's/# ru_RU.UTF-8 UTF-8/ru_RU.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG=ru_RU.UTF-8 LC_ALL=ru_RU.UTF-8

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Uvicorn слушает порт из окружения (Timeweb может пробрасывать свои порты)
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
