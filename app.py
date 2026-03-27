import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

STATUS_COLORS = {0: "🟢", 1: "🟡", 2: "🔴"}
STATUS_LABELS = {0: "Healthy", 1: "Warning", 2: "Critical Failure"}
STATUS_MESSAGES = {
    0: "Machine operating normally. No action required.",
    1: "Anomaly detected. Schedule maintenance soon.",
    2: "CRITICAL: Machine failure imminent. Stop immediately!"
}

st.set_page_config(page_title="Predictive Maintenance", page_icon="🏭")
st.title("🏭 Predictive Maintenance Dashboard")
st.caption("Real-time machine health monitoring")

st.sidebar.header("Sensor Readings")
temperature = st.sidebar.slider("Temperature (°C)", 50.0, 150.0, 70.0)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.1, 2.5, 0.5)
pressure = st.sidebar.slider("Pressure (bar)", 70.0, 200.0, 100.0)
rpm = st.sidebar.slider("RPM", 1000.0, 2500.0, 1500.0)
humidity = st.sidebar.slider("Humidity (%)", 40.0, 80.0, 60.0)

input_data = pd.DataFrame([[temperature, vibration, pressure, rpm, humidity]],
    columns=["temperature", "vibration", "pressure", "rpm", "humidity"])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Health Score", f"{probabilities[0]*100:.1f}%", "Healthy")
with col2:
    st.metric("Warning Risk", f"{probabilities[1]*100:.1f}%", "Warning")
with col3:
    st.metric("Failure Risk", f"{probabilities[2]*100:.1f}%", "Critical")

st.markdown("---")
status_icon = STATUS_COLORS[prediction]
status_label = STATUS_LABELS[prediction]
status_msg = STATUS_MESSAGES[prediction]

if prediction == 0:
    st.success(f"{status_icon} {status_label} — {status_msg}")
elif prediction == 1:
    st.warning(f"{status_icon} {status_label} — {status_msg}")
else:
    st.error(f"{status_icon} {status_label} — {status_msg}")

st.markdown("---")
st.subheader("Current Sensor Values")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Temperature", f"{temperature}°C")
col2.metric("Vibration", f"{vibration} mm/s")
col3.metric("Pressure", f"{pressure} bar")
col4.metric("RPM", f"{rpm}")
col5.metric("Humidity", f"{humidity}%")

st.markdown("---")
st.subheader("Simulate Machine Deterioration")
if st.button("Run Simulation"):
    progress = st.progress(0)
    status_placeholder = st.empty()
    for i in range(100):
        temp = 70 + (i * 0.8)
        vib = 0.5 + (i * 0.018)
        pres = 100 + (i * 0.6)
        r = 1500 + (i * 8)
        hum = 60.0
        data = pd.DataFrame([[temp, vib, pres, r, hum]],
            columns=["temperature", "vibration", "pressure", "rpm", "humidity"])
        pred = model.predict(scaler.transform(data))[0]
        icon = STATUS_COLORS[pred]
        label = STATUS_LABELS[pred]
        status_placeholder.markdown(f"**Step {i+1}/100** — {icon} {label} | Temp: {temp:.1f}°C | Vibration: {vib:.3f}")
        progress.progress(i + 1)
        time.sleep(0.05)
    st.success("Simulation complete! Watch how status changes from Healthy → Warning → Critical")