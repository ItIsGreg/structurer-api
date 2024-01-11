FROM python:3.12

WORKDIR /structurer-api

RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-deu

COPY ./requirements.txt /structurer-api/requirements.txt

RUN python3 -m pip install -r requirements.txt --no-cache-dir

COPY ./data /structurer-api/data

COPY ./src /structurer-api/src

WORKDIR /structurer-api/src

EXPOSE 8000

CMD ["uvicorn", "structurer_api.main:app", "--host", "0.0.0.0", "--port", "8000"]