import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import yaml

def load_processed_data():
    """Carga los datos procesados desde la carpeta data/processed."""
    train_path = "../data/processed/processed_train.csv"
    df_train = pd.read_csv(train_path)
    return df_train

def train_model(df_train):
    """Entrena un modelo de RandomForest y devuelve el modelo entrenado."""
    X = df_train.drop("price_range", axis=1)
    y = df_train["price_range"]
    
    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar un modelo RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_val, y_train, y_val

def save_model(model, model_name):
    """Guarda el modelo entrenado en formato pickle en la carpeta models."""
    model_path = f"../models/{model_name}.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

def save_model_config(model, config_file):
    """Guarda la configuración del modelo en un archivo YAML."""
    config = {
        'n_estimators': model.n_estimators,
        'random_state': model.random_state,
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf,
        'bootstrap': model.bootstrap
    }
    
    with open(config_file, 'w') as file:
        yaml.dump(config, file)

def main():
    # Cargar datos procesados
    df_train = load_processed_data()
    
    # Entrenar modelo
    model, X_train, X_val, y_train, y_val = train_model(df_train)
    
    # Guardar modelo entrenado con identificador `n`
    model_id = 1  # Esto puede cambiar dependiendo de cómo lo manejes (e.g., un contador)
    save_model(model, f"trained_model_{model_id}")
    
    # Guardar configuración del modelo
    save_model_config(model, "../models/model_config.yaml")
    
    # Guardar modelo final
    save_model(model, "final_model")
    
    # Guardar datasets utilizados en el entrenamiento
    X_train.to_csv("../data/train/X_train.csv", index=False)
    X_val.to_csv("../data/test/X_val.csv", index=False)
    y_train.to_csv("../data/train/y_train.csv", index=False)
    y_val.to_csv("../data/test/y_val.csv", index=False)

if __name__ == "__main__":
    main()
