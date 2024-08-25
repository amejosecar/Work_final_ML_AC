import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo final
model = joblib.load('../models/final_model.pkl')

# Definir la función para hacer predicciones
def predict(features):
    # Asegurarse de que las características estén en el formato correcto
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Configuración de la aplicación
st.title("Predicción de Price Range con el Modelo Final")
st.write("""
### Introduce las características del dispositivo para predecir su rango de precios:
""")

# Interfaz de usuario para recibir entradas
battery_power = st.number_input('Battery Power', min_value=0, max_value=5000, value=1500)
blue = st.selectbox('Bluetooth', options=[0, 1], help='0: No tiene Bluetooth, 1: Tiene Bluetooth')
clock_speed = st.slider('Clock Speed', min_value=0.5, max_value=3.0, step=0.1, value=1.5)
dual_sim = st.selectbox('Dual SIM', options=[0, 1], help='0: No tiene Dual SIM, 1: Tiene Dual SIM')
fc = st.slider('Front Camera (MP)', min_value=0, max_value=20, value=5)
four_g = st.selectbox('4G', options=[0, 1], help='0: No tiene 4G, 1: Tiene 4G')
int_memory = st.slider('Internal Memory (GB)', min_value=0, max_value=256, value=64)
m_dep = st.slider('Mobile Depth (cm)', min_value=0.1, max_value=1.0, step=0.1, value=0.5)
mobile_wt = st.slider('Mobile Weight (g)', min_value=80, max_value=250, value=150)
n_cores = st.slider('Number of Cores', min_value=1, max_value=8, value=4)
pc = st.slider('Primary Camera (MP)', min_value=0, max_value=20, value=12)
px_height = st.slider('Pixel Resolution Height', min_value=0, max_value=2560, value=1280)
px_width = st.slider('Pixel Resolution Width', min_value=0, max_value=2560, value=720)
ram = st.slider('RAM (MB)', min_value=512, max_value=8192, step=256, value=4096)
sc_h = st.slider('Screen Height (cm)', min_value=0, max_value=20, value=14)
sc_w = st.slider('Screen Width (cm)', min_value=0, max_value=20, value=7)
talk_time = st.slider('Talk Time (hours)', min_value=0, max_value=20, value=10)
three_g = st.selectbox('3G', options=[0, 1], help='0: No tiene 3G, 1: Tiene 3G')
touch_screen = st.selectbox('Touch Screen', options=[0, 1], help='0: No tiene pantalla táctil, 1: Tiene pantalla táctil')
wifi = st.selectbox('WiFi', options=[0, 1], help='0: No tiene WiFi, 1: Tiene WiFi')

# Recoger todas las entradas en una lista
features = [
    battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt,
    n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi
]

# Botón de predicción
if st.button('Predecir Rango de Precio'):
    prediction = predict(features)
    st.success(f'El modelo predice que el rango de precios es: {prediction}')

# Mostrar información sobre el rango de precios
st.write("""
### Rango de Precios:
- 0: Low cost
- 1: Medium cost
- 2: High cost
- 3: Very high cost
""")
