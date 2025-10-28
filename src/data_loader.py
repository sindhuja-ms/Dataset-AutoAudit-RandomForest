import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path="data/weather_classification_data.csv", target_col="Weather Type"):
    df = pd.read_csv(path)
    df = df.dropna()

    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    X = df[numeric_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, le
