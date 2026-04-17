FROM python:3.12

WORKDIR /app

COPY MLProject /app/MLProject

RUN pip install mlflow==2.19.0 scikit-learn pandas matplotlib seaborn

WORKDIR /app/MLProject

CMD ["python", "modelling.py"]
