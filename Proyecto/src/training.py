import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(input_path, model_output_path):
    # Cargar los datos procesados
    df = pd.read_csv(os.path.join(input_path, 'df_train_proce.csv'))

    # Separación de características y variable objetivo
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento del modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(model, os.path.join(model_output_path, 'random_forest_model.pkl'))
    print(f"Modelo guardado en {model_output_path}")

    # Guardar los datasets de entrenamiento y prueba
    X_train.to_csv(os.path.join(model_output_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(model_output_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(model_output_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(model_output_path, 'y_test.csv'), index=False)
    print(f"Datasets guardados en {model_output_path}")

if __name__ == "__main__":
    input_path = "../data/processed"
    model_output_path = "../data/train"
    train_model(input_path, model_output_path)
