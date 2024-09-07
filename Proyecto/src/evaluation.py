import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def load_test_data():
    """Carga los datos de prueba desde la carpeta data/test."""
    X_val = pd.read_csv("../data/test/X_val.csv")
    y_val = pd.read_csv("../data/test/y_val.csv")
    return X_val, y_val

def load_model(model_name):
    """Carga el modelo entrenado desde la carpeta models."""
    model_path = f"../models/{model_name}.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_val, y_val):
    """Genera un informe de clasificación y matriz de confusión."""
    y_pred = model.predict(X_val)
    
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

def main():
    # Cargar datos de prueba
    X_val, y_val = load_test_data()
    
    # Cargar modelo
    model = load_model("trained_model")
    
    # Evaluar modelo
    evaluate_model(model, X_val, y_val)

if __name__ == "__main__":
    main()
