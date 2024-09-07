import pandas as pd
<<<<<<< HEAD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

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

def main():
    # Cargar datos procesados
    df_train = load_processed_data()
    
    # Entrenar modelo
    model, X_train, X_val, y_train, y_val = train_model(df_train)
    
    # Guardar modelo entrenado
    save_model(model, "trained_model")
    
    # Guardar datasets utilizados en el entrenamiento
    X_train.to_csv("../data/train/X_train.csv", index=False)
    X_val.to_csv("../data/test/X_val.csv", index=False)
    y_train.to_csv("../data/train/y_train.csv", index=False)
    y_val.to_csv("../data/test/y_val.csv", index=False)


if __name__ == "__main__":
    main()
=======
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle  # Importa pickle para guardar el escalador y los modelos

def train_models(df_train, df_test_proce):
    # Separación de características y target
    X = df_train.drop('price_range', axis=1)
    y = df_train['price_range']

    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Guardar los datasets de entrenamiento y validación
    pd.DataFrame(X_train).to_csv("../data/train/X_train.csv", index=False)
    pd.DataFrame(X_val).to_csv("../data/train/X_val.csv", index=False)
    pd.DataFrame(y_train).to_csv("../data/train/y_train.csv", index=False)
    pd.DataFrame(y_val).to_csv("../data/train/y_val.csv", index=False)

    # Preparación de los datos de test
    X_test_scaled = scaler.transform(df_test_proce)

    # Entrenamiento y evaluación de modelos
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "SVC": SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, solver='liblinear', random_state=42)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f"{model_name} - Accuracy on Validation:", accuracy_score(y_val, y_pred))
    
    # Guardar los modelos entrenados
    for i, (model_name, model) in enumerate(models.items(), start=1):
        with open(f"../models/trained_model_{i}.pkl", 'wb') as f:
            pickle.dump(model, f)

    # Guardar el modelo final (elige el modelo que quieras como final, por ejemplo el RandomForest)
    final_model = models["Random Forest"]  # Ajusta esto según el modelo final que elijas
    with open("../models/final_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)

    # Guardar el escalador
    with open("../models/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    return models, X_val, y_val, X_test_scaled

if __name__ == "__main__":
    df_train = pd.read_csv("../data/train/train.csv")
    df_test_proce = pd.read_csv("../data/processed/df_test_proce.csv")
    models, X_val, y_val, X_test_scaled = train_models(df_train, df_test_proce)
>>>>>>> 203a8178865fdf70dff0554c3b8c102453247f7e
