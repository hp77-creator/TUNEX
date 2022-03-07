FROM python:3.8

RUN dpkg --add-architecture i386 && apt update
RUN apt-get clean
RUN apt install -y libgl1-mesa-dev

WORKDIR /app

COPY . /app

RUN pip --no-cache-dir install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["python"]

CMD ["manage.py", "runserver"]
