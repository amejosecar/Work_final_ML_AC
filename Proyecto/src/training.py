import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

def train_model():
    # Cargar los datos procesados
    df_train = pd.read_csv("../data/processed/df_train_processed.csv")
    df_test = pd.read_csv("../data/processed/df_test_processed.csv")

    # Separar características y target
    X = df_train.drop('price_range', axis=1)
    y = df_train['price_range']

    # Escalar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(df_test)

    # División de los datos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    os.makedirs("../models", exist_ok=True)
    with open("../models/random_forest_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    # Guardar los datasets utilizados en el entrenamiento
    pd.DataFrame(X_train).to_csv("../data/processed/X_train.csv", index=False)
    pd.DataFrame(X_val).to_csv("../data/processed/X_val.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv("../data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv", index=False)
    pd.DataFrame(y_val).to_csv("../data/processed/y_val.csv", index=False)

if __name__ == "__main__":
    train_model()
