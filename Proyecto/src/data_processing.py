import pandas as pd
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
