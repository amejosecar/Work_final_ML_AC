import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def train_and_save_models():
    # Cargar un conjunto de datos de ejemplo
    data = load_iris()
    X = data.data
    y = data.target

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir y entrenar los modelos
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/trained_model_rf.pkl')

    svc = SVC(probability=True, kernel='linear', random_state=42)
    svc.fit(X_train, y_train)
    joblib.dump(svc, 'models/trained_model_svc.pkl')

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    joblib.dump(gb, 'models/trained_model_gb.pkl')

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'models/trained_model_lr.pkl')

    # Guardar el modelo final (puedes elegir uno de los anteriores como final)
    final_model = rf  # Suponiendo que seleccionamos RandomForestClassifier como el modelo final
    joblib.dump(final_model, 'models/final_model.pkl')

    # Guardar la configuración del modelo final
    model_config = {
        'model': 'RandomForestClassifier',
        'params': {
            'n_estimators': 100,
            'random_state': 42
        }
    }
    with open('models/model_config.yaml', 'w') as file:
        yaml.dump(model_config, file)

    print("Modelos y configuración guardados exitosamente.")

if __name__ == "__main__":
    train_and_save_models()
