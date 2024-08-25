import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_data():
    # Cargar datos
    df_train = pd.read_csv('data/raw/train.csv')
    df_test = pd.read_csv('data/raw/test.csv')

    # Separación de características y target en el conjunto de entrenamiento
    X_train = df_train.drop('price_range', axis=1)
    y_train = df_train['price_range']

    # Eliminar la columna 'id' de df_train y df_test si está presente
    if 'id' in X_train.columns:
        X_train = X_train.drop('id', axis=1)
    if 'id' in df_test.columns:
        df_test = df_test.drop('id', axis=1)

    # El conjunto de prueba debe tener las mismas características que el conjunto de entrenamiento
    X_test = df_test.copy()

    # Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar los datos procesados
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed/X_train_scaled.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed/X_test_scaled.csv', index=False)

    # Guardar el target para el test set si es necesario (asumiendo que test.csv tiene 'price_range')
    # Como mencionaste que 'test.csv' es solo para la predicción, puede que no tenga 'price_range'
    # Por lo tanto, se podría omitir esta línea si no hay 'price_range' en test.csv
    if 'price_range' in df_test.columns:
        pd.DataFrame(df_test['price_range']).to_csv('data/processed/y_test.csv', index=False)

    print("Datos procesados y guardados exitosamente.")

if __name__ == "__main__":
    process_data()
