import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import pickle

def evaluate_model():
    # Cargar el modelo entrenado
    with open("../models/random_forest_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    # Cargar los datos de prueba procesados
    X_test = pd.read_csv("../data/processed/X_test.csv")
    
    # Predecir utilizando el modelo entrenado
    y_test_pred = model.predict(X_test)

    # Como no tenemos las etiquetas de test, se pueden utilizar métricas de validación o cruzar con datos adicionales si están disponibles.

    print("Evaluación del modelo:")
    # Si tuvieras etiquetas de test:
    # y_test = pd.read_csv("../data/processed/y_test.csv")
    # print("Accuracy on Test:", accuracy_score(y_test, y_test_pred))
    # print(classification_report(y_test, y_test_pred))
    
    # Aquí sólo se muestran las predicciones porque no tenemos etiquetas de test.
    print(y_test_pred)

if __name__ == "__main__":
    evaluate_model()
