FROM python:3.8.5

WORKDIR /usr/src/app

COPY . .
RUN python -m pip --no-cache-dir install .

CMD ["python", "./cli.py"]
