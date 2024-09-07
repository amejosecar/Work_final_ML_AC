import streamlit as st
import pandas as pd
import pickle
import os

#st.write('Entre al portal aap.py')

# Definir la ruta relativa a la ubicación de app.py
model_path = "./models"
print(model_path)
scaler_file = os.path.join(model_path, "scaler.pkl")
final_model_file = os.path.join(model_path, "final_model.pkl")

# Verificar que los archivos de modelo existen
if not os.path.exists(final_model_file) or not os.path.exists(scaler_file):
    st.error(f"Los archivos de modelo o scaler no existen. Verifica las rutas:\n- {final_model_file}\n- {scaler_file}")
    st.stop()

# Cargar el modelo final y el scaler
try:
    with open(final_model_file, 'rb') as f:
        #st.write('entre en el 1er with')
        final_model = pickle.load(f)
    with open(scaler_file, 'rb') as f:
        #st.write('entre en el 2do with')
        scaler = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"No se pudo cargar el archivo: {e}")
    st.stop()

# Definir las características esperadas
expected_features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 
                      'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 
                      'pc', 'ram', 'talk_time', 'three_g', 'touch_screen', 
                      'wifi', 'px_height', 'px_width', 'sc_h', 'sc_w']

# Streamlit app
st.title('Phone Price Range Prediction')

# Input from user
user_data = {feature: st.number_input(feature, min_value=0, value=0) for feature in expected_features}
user_df = pd.DataFrame([user_data])

# Asegúrate de que el DataFrame de entrada tenga las mismas columnas en el mismo orden
user_df = user_df[expected_features]

# Escalar las características
user_data_scaled = scaler.transform(user_df)

# Realizar la predicción
prediction = final_model.predict(user_data_scaled)

# Mostrar el resultado
st.write('Predicted Price Range:', prediction[0])
