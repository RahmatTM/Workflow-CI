FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install mlflow pandas numpy scikit-learn matplotlib seaborn

CMD ["python", "modelling.py"]