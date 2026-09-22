import streamlit as st
import numpy as np
import tensorflow as tf

st.title("Machine Temperature Predictor")

# Load trained RNN model
model = tf.keras.models.load_model("machine_temperature_rnn.keras")

# Previous timestamp 1
temp1 = st.number_input("Previous Temperature 1", value=81.0)
vib1 = st.number_input("Previous Vibration 1", value=3.5)

# Previous timestamp 2
temp2 = st.number_input("Previous Temperature 2", value=83.0)
vib2 = st.number_input("Previous Vibration 2", value=3.6)

if st.button("Predict Next Temperature"):

    input_data = np.array([
        [[temp1, vib1],
         [temp2, vib2]]
    ])

    prediction = model.predict(input_data, verbose=0)[0][0]

    st.success(
        f"Predicted Next Temperature: {prediction:.2f} °C"
    )
