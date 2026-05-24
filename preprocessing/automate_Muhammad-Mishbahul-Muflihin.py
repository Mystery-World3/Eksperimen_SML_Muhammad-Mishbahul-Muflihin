import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'telco_raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    prep_dir = os.path.join(base_dir, 'preprocessing', 'telco_preprocessing')

    df = pd.read_csv(raw_path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    y_train_df = pd.DataFrame(y_train, columns=['Churn'])
    y_test_df = pd.DataFrame(y_test, columns=['Churn'])

    os.makedirs(prep_dir, exist_ok=True)

    X_train_df.to_csv(os.path.join(prep_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(prep_dir, 'X_test.csv'), index=False)
    y_train_df.to_csv(os.path.join(prep_dir, 'y_train.csv'), index=False)
    y_test_df.to_csv(os.path.join(prep_dir, 'y_test.csv'), index=False)

if __name__ == "__main__":
    run_preprocessing()