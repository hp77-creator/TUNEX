FROM python:3.8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1
RUN dpkg --add-architecture i386 && apt update && apt-get clean && apt install -y libgl1-mesa-dev && pip install --upgrade pip
WORKDIR /app
COPY requirements.txt /app/
RUN pip --no-cache-dir install -r requirements.txt
COPY . /app/

