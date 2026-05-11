import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# ENABLE MLFLOW AUTOLOG
mlflow.sklearn.autolog()

# LOAD DATASET
url = 'https://media.githubusercontent.com/media/RahmatTM/Workflow-CI/refs/heads/main/MLProject/dataset_preprocessing/data_clean.csv'

df = pd.read_csv(url)

# FEATURE & TARGET
target_column = 'price_category'

X = df.drop(columns=[target_column])
y = df[target_column]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# START MLFLOW RUN
with mlflow.start_run():

    # TRAIN MODEL
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

# PREDICTION
y_pred = model.predict(X_test)

# EVALUATION
accuracy = accuracy_score(y_test, y_pred)

print(f'Akurasi: {accuracy}')

print('\nClassification Report')
print(classification_report(y_test, y_pred))

print('Training selesai')