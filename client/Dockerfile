FROM python:3.10

WORKDIR /client

ADD . /client

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD [ "Python3", "client.py" ]