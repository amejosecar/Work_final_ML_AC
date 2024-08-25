import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_score, n_classes, label):
    plt.figure(figsize=(10, 8))
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {label}')
    plt.legend(loc='lower right')
    plt.show()

def evaluate_model():
    # Cargar datos de prueba
    X_test = pd.read_csv('data/processed/X_test_scaled.csv')
    try:
        y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
        y_test = y_test.values
    except FileNotFoundError:
        y_test = None

    # Convertir a matrices numpy
    X_test = X_test.values

    # Cargar modelos
    rf = joblib.load('models/random_forest_model.pkl')
    svc = joblib.load('models/svc_model.pkl')
    gb = joblib.load('models/gradient_boosting_model.pkl')
    lr = joblib.load('models/logistic_regression_model.pkl')

    # Predicciones y probabilidades
    y_pred_rf = rf.predict(X_test)
    y_score_rf = rf.predict_proba(X_test)
    y_pred_svc = svc.predict(X_test)
    y_score_svc = svc.predict_proba(X_test)
    y_pred_gb = gb.predict(X_test)
    y_score_gb = gb.predict_proba(X_test)
    y_pred_lr = lr.predict(X_test)
    y_score_lr = lr.predict_proba(X_test)

    # Si y_test está disponible, imprimir matrices de confusión y reportes de clasificación
    if y_test is not None:
        # Matriz de Confusión
        print("Random Forest Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_rf))
        print("Support Vector Classifier Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_svc))
        print("Gradient Boosting Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_gb))
        print("Logistic Regression Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_lr))

        # Curvas ROC
        n_classes = len(np.unique(y_test))
        plot_roc_curve(y_test, y_score_rf, n_classes, 'Random Forest')
        plot_roc_curve(y_test, y_score_svc, n_classes, 'SVC')
        plot_roc_curve(y_test, y_score_gb, n_classes, 'Gradient Boosting')
        plot_roc_curve(y_test, y_score_lr, n_classes, 'Logistic Regression')

        # Reporte de clasificación
        print("Random Forest Classification Report:")
        print(classification_report(y_test, y_pred_rf))
        print("Support Vector Classifier Classification Report:")
        print(classification_report(y_test, y_pred_svc))
        print("Gradient Boosting Classification Report:")
        print(classification_report(y_test, y_pred_gb))
        print("Logistic Regression Classification Report:")
        print(classification_report(y_test, y_pred_lr))
    else:
        print("No se encontró 'y_test.csv'. Se omiten las métricas que requieren etiquetas verdaderas.")

if __name__ == "__main__":
    evaluate_model()
