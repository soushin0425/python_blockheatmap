FROM python:3.10

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r /app/requirements.txt

CMD [ "bash" ]