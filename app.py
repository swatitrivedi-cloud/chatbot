import streamlit as st
import pandas as pd
import google.genai as genai

# -----------------------------
# GEMINI SETUP (CORRECT FOR SDK 2.10.0)
# -----------------------------
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def llm_explain(prompt):
    response = client.generate_text(
        model="gemini-1.5-flash-latest",
        prompt=prompt
    )
    return response.text

# -----------------------------
# LOAD DATA
# -----------------------------
CSV_PATH = "data/window_features_1min.csv"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

df = load_data()

# -----------------------------
# BACKEND FUNCTIONS
# -----------------------------
def get_voltage_summary(device):
    d = df[df['sensorDeviceId'] == device]
    return {
        "device": device,
        "avg_voltage": round(d['volt_mean'].mean(), 2),
        "min_voltage": round(d['volt_mean'].min(), 2),
        "max_voltage": round(d['volt_mean'].max(), 2),
        "high_spread_events": int((d['spread_max'] > 3).sum())
    }

def get_frequency_summary(device):
    d = df[df['sensorDeviceId'] == device]
    return {
        "device": device,
        "avg_frequency": round(d['freq_mean'].mean(), 4),
        "min_frequency": round(d['freq_mean'].min(), 4),
        "max_frequency": round(d['freq_mean'].max(), 4),
        "frequency_deviation_events": int(((d['freq_mean'] < 49.8) | (d['freq_mean'] > 50.2)).sum()),
        "frequency_instability_events": int((d['freq_std'] > 0.02).sum())
    }

def get_anomaly_summary(device):
    d = df[df['sensorDeviceId'] == device]
    return {
        "device": device,
        "total_anomalies": int((d['anomalies'] != "Normal").sum())
    }

def get_missing_data_report(device):
    d = df[df['sensorDeviceId'] == device]
    return {
        "device": device,
        "missing_windows": int((d['data_quality'] == "Missing").sum()),
        "partial_windows": int((d['data_quality'] == "Partial").sum()),
        "good_windows": int((d['data_quality'] == "Good").sum())
    }

def classify_question(q):
    q = q.lower()
    if any(k in q for k in ["voltage", "volt", "imbalance"]):
        return "voltage"
    if any(k in q for k in ["frequency", "hz", "freq"]):
        return "frequency"
    if any(k in q for k in ["anomaly", "spike", "abnormal"]):
        return "anomalies"
    if any(k in q for k in ["missing", "gap", "incomplete"]):
        return "missing_data"
    if any(k in q for k in ["overall", "health", "performance"]):
        return "multi_topic"
    return "conversational"

def router(question, device):
    category = classify_question(question)

    if category == "voltage":
        return llm_explain(str(get_voltage_summary(device)))
    if category == "frequency":
        return llm_explain(str(get_frequency_summary(device)))
    if category == "anomalies":
        return llm_explain(str(get_anomaly_summary(device)))
    if category == "missing_data":
        return llm_explain(str(get_missing_data_report(device)))
    if category == "multi_topic":
        combined = {
            "voltage": get_voltage_summary(device),
            "frequency": get_frequency_summary(device),
            "anomalies": get_anomaly_summary(device),
            "missing_data": get_missing_data_report(device)
        }
        return llm_explain(str(combined))

    return llm_explain(f"User said: {question}")

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("⚡ Energy Dataset Chatbot (Gemini + Streamlit)")
st.write("Ask anything about voltage, frequency, anomalies, or missing data.")

device = st.selectbox("Select device:", df["sensorDeviceId"].unique())

user_question = st.text_input("Your question:")

if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        answer = router(user_question, device)
        st.success(answer)

