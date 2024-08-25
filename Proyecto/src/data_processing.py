import pandas as pd
import os

def process_data():
    # Cargar los datos crudos
    df_train = pd.read_csv("../data/raw/train.csv")
    df_test = pd.read_csv("../data/raw/test.csv")
    
    # Eliminar el campo 'id' del dataset de test
    if 'id' in df_test.columns:
        df_test = df_test.drop(columns=['id'])
    
    # Guardar los datos procesados
    os.makedirs("../data/processed", exist_ok=True)
    df_train.to_csv("../data/processed/df_train_processed.csv", index=False)
    df_test.to_csv("../data/processed/df_test_processed.csv", index=False)

if __name__ == "__main__":
    process_data()

