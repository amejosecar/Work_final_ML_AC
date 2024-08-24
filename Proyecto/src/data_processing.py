import pandas as pd
import os

def process_data(input_path, output_path):
    # Cargar los datos en bruto
    df_raw = pd.read_csv(os.path.join(input_path, 'train.csv'))

    # Procesamiento de datos (ejemplo: normalización, eliminación de NaN, etc.)
    df_processed = df_raw.dropna()  # Ejemplo sencillo, elimina filas con NaN

    # Guardar los datos procesados
    df_processed.to_csv(os.path.join(output_path, 'df_train_proce.csv'), index=False)
    print(f"Datos procesados guardados en {output_path}")

if __name__ == "__main__":
    input_path = "../data/raw"
    output_path = "../data/processed"
    process_data(input_path, output_path)
