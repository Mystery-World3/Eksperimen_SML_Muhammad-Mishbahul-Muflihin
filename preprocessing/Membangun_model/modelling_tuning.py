import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='Mystery-World3', repo_name='MLOps-Telco-Churn', mlflow=True)

X_train = pd.read_csv('telco_preprocessing/X_train.csv')
y_train = pd.read_csv('telco_preprocessing/y_train.csv').values.ravel()
X_test = pd.read_csv('telco_preprocessing/X_test.csv')
y_test = pd.read_csv('telco_preprocessing/y_test.csv').values.ravel()

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

with mlflow.start_run():
    print("Melatih model dan mencari parameter terbaik...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mlflow.log_params(grid_search.best_params_)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    mlflow.log_metrics(metrics)
    
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importances.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    print(f"Training selesai! Model berhasil dilog ke DagsHub.")
    print(f"Akurasi: {metrics['accuracy']:.4f}")