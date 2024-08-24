import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def evaluate_model(model_path, test_data_path):
    # Cargar el modelo entrenado
    model = joblib.load(os.path.join(model_path, 'random_forest_model.pkl'))

    # Cargar los datos de prueba
    X_test = pd.read_csv(os.path.join(test_data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(test_data_path, 'y_test.csv'))

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Mostrar resultados
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    model_path = "../data/train"
    test_data_path = "../data/train"
    evaluate_model(model_path, test_data_path)
