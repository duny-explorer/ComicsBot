FROM ubuntu:latest

MAINTAINER Daria Denisova 'duny.explorer@yandex.ru'

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

COPY ./app
WORKDIR /app

RUN python -m pip install -r requirements.txt
RUN pip install https://github.com/aiogram/aiogram/archive/refs/heads/dev-3.x.zip

ENTRYPOINT ["python"]
CMD [ "bot/main.py" ]
