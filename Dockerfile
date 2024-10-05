# Используем базовый образ Ubuntu
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Устанавливаем рабочую директорию
WORKDIR /usr/src/app

# Копируем файлы с зависимостями и устанавливаем их
#RUN pip install --ignore-installed blinker

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src .

# Открываем порт
EXPOSE 8090

# Запускаем приложение
CMD ["python", "./main.py"]
