import pandas as pd

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
