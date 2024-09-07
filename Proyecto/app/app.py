import streamlit as st
import pickle
import pandas as pd

def load_model():
    """Carga el modelo final guardado en la carpeta models."""
    model_path = "./models/final_model.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def predict(model, input_data):
    """Realiza una predicción utilizando el modelo entrenado."""
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Predicción de rango de precio de móviles")
    
    # Cargar el modelo
    model = load_model()
    
    # Crear entradas para las características del modelo
    battery_power = st.number_input("Battery Power", min_value=500, max_value=2000, value=1000)
    clock_speed = st.number_input("Clock Speed", min_value=0.5, max_value=3.0, value=1.5)
    # Puedes añadir más inputs según sea necesario
    
    # Realizar predicción al hacer clic en el botón
    if st.button("Predecir"):
        input_data = pd.DataFrame({
            "battery_power": [battery_power],
            "clock_speed": [clock_speed]
            # Añadir más características si es necesario
        })
        
        prediction = predict(model, input_data)
        st.write(f"Predicción del rango de precio: {prediction}")

if __name__ == "__main__":
    main()
