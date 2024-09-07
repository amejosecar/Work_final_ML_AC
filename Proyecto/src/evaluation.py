import pandas as pd
<<<<<<< HEAD
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
=======
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
import pickle

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_score, n_classes, label):
    plt.figure(figsize=(12, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (class {i}) AUC = {roc_auc:.2f}')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {label}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def evaluate_model(model, X_val, y_val):
    y_bin = label_binarize(y_val, classes=[0, 1, 2, 3])
    n_classes = y_bin.shape[1]

    y_pred = model.predict(X_val)
    y_score = model.predict_proba(X_val)
    
    print(f"Classification Report:\n", classification_report(y_val, y_pred))
    print(f"Accuracy on Validation:", accuracy_score(y_val, y_pred))
    plot_roc_curve(y_bin, y_score, n_classes, "Final Model")
    plot_confusion_matrix(y_val, y_pred, 'Final Model Confusion Matrix')

if __name__ == "__main__":
    # Cargar el modelo final
    with open("../models/final_model.pkl", 'rb') as f:
        final_model = pickle.load(f)
    
    X_val = pd.read_csv("../data/train/X_val.csv")
    y_val = pd.read_csv("../data/train/y_val.csv").squeeze()

    evaluate_model(final_model, X_val, y_val)
>>>>>>> 203a8178865fdf70dff0554c3b8c102453247f7e
