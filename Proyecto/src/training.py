import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_models():
    # Cargar datos procesados
    X_train = pd.read_csv('data/processed/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()

    # Dividir el conjunto de entrenamiento en entrenamiento y validación
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Inicializar modelos
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svc = SVC(probability=True, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    # Entrenar modelos
    rf.fit(X_train_split, y_train_split)
    svc.fit(X_train_split, y_train_split)
    gb.fit(X_train_split, y_train_split)
    lr.fit(X_train_split, y_train_split)

    # Guardar modelos
    joblib.dump(rf, 'models/random_forest_model.pkl')
    joblib.dump(svc, 'models/svc_model.pkl')
    joblib.dump(gb, 'models/gradient_boosting_model.pkl')
    joblib.dump(lr, 'models/logistic_regression_model.pkl')

    print("Modelos entrenados y guardados exitosamente.")

if __name__ == "__main__":
    train_models()
