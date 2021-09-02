# syntax=docker/dockerfile:1

FROM python:3

WORKDIR /badge-detection

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
CMD ["export", "PYTHONPATH='${PYTHONPATH}:/`pwd`'"]
CMD [ "python3", "app/interface/main.py"]