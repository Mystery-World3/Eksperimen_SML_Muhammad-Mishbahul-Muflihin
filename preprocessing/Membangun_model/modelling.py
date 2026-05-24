import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

X_train = pd.read_csv('telco_preprocessing/X_train.csv')
y_train = pd.read_csv('telco_preprocessing/y_train.csv').values.ravel()

mlflow.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Basic training log selesai!")