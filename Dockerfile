FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install mlflow pandas scikit-learn

CMD ["python", "MLProject/modelling.py"]