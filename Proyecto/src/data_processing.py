import pandas as pd
<<<<<<< HEAD
import os

def load_raw_data():
    """Carga los datos en bruto desde la carpeta data/raw."""
    train_path = "../data/raw/train.csv"
    test_path = "../data/raw/test.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_test

def process_data(df):
    """Realiza el procesamiento básico del DataFrame."""
    # Eliminar filas con valores faltantes (puedes modificar según lo necesario)
    df = df.dropna()
    
    # Si es necesario, añade otras transformaciones aquí
    # Por ejemplo, normalización, codificación, etc.
    
    return df

def save_processed_data(df_train, df_test):
    """Guarda los datos procesados en la carpeta data/processed."""
    processed_train_path = "../data/processed/processed_train.csv"
    processed_test_path = "../data/processed/processed_test.csv"
    
    df_train.to_csv(processed_train_path, index=False)
    df_test.to_csv(processed_test_path, index=False)

def main():
    # Cargar los datos
    df_train, df_test = load_raw_data()
    
    # Procesar los datos
    df_train = process_data(df_train)
    df_test = process_data(df_test)
    
    # Guardar los datos procesados
    save_processed_data(df_train, df_test)

if __name__ == "__main__":
    main()
=======

def load_and_process_data():
    # Cargar los datos en bruto
    df_train = pd.read_csv("../data/raw/train.csv")
    df_test = pd.read_csv("../data/raw/test.csv")

    # Verificar datos nulos
    print("Datos nulos en df_train:\n", df_train.isna().sum())
    print("Datos nulos en df_test:\n", df_test.isna().sum())

    # Procesar df_test eliminando la columna "id"
    df_test_proce = df_test.drop(columns=["id"])

    # Guardar el DataFrame procesado en la carpeta data/processed
    df_test_proce.to_csv('../data/processed/df_test_proce.csv', index=False)

    return df_train, df_test_proce

if __name__ == "__main__":
    df_train, df_test_proce = load_and_process_data()
>>>>>>> 203a8178865fdf70dff0554c3b8c102453247f7e
