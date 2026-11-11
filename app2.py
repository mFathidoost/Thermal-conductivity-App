import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(
    page_title="Thermal Conductivity Predictor",
    page_icon="🔬",
    layout="centered"
)

@st.cache_resource
def load_resources():
    # Load trained model
    model = joblib.load("gb_model.pkl")
    
    # Load dataset to recreate scalers on the fly
    data = pd.read_excel('total_data_KR10_70_Rs.xlsx', engine='openpyxl')
    data = data.iloc[:, 0:12].copy()
    data['AR'] = None
    data['vol'] = None

    for i in range(len(data)):
        a = str(data.iloc[i, 2])
        b = int(a[-1])
        c = int(a[:-1])
        data.iloc[i, 10] = b
        data.iloc[i, 11] = c

    data.columns = [
        'KR', 'Rs', 'AR_vol', 'Unnamed: 3', 'k', 'k_egv', 
        'k_eff_ratio', 'k_std', 'k_max', 'k_min', 'AR', 'V_f'
    ]

    X = data[['Rs', 'AR', 'V_f', 'KR']].astype(float)
    y = data['k_eff_ratio'].astype(float)

    # Replicate train/test split and scaler fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    X_scaler = MinMaxScaler()
    X_scaler.fit(X_train)

    y_scaler = MinMaxScaler()
    y_scaler.fit(np.array(y_train).reshape(-1, 1))

    return model, X_scaler, y_scaler

model, X_scaler, y_scaler = load_resources()

# --- App Header & Description ---
st.title("🔬 Thermal Conductivity Predictor")
st.markdown("""
This web application leverages a data-driven **machine learning model** trained on computational thermal homogenization simulation data. It predicts the effective thermal conductivity ratio ($k_{eff} / k_{(i)}$) of ellipsoidal composite microstructure systems instantly based on structural and material parameters.
""")

# --- Add Your Image Here ---
# Replace 'microstructure.png' with your actual image filename and caption
st.image("micro.png", caption="Representative Ellipsoidal Composite Microstructure")

with st.expander("ℹ️ About the Input Parameters"):
    st.markdown("""
    * **$R_s$ (Interfacial Resistance):** Thermal boundary resistance at phase interfaces.
    * **$A_r$ (Aspect Ratio):** Shape parameter characterizing microstructural inclusions.
    * **$V_f$ (Volume Fraction):** Concentration/volume percentage of the dispersed phase.
    * **$K_r$ (Conductivity Ratio):** Ratio of constituent phase thermal conductivities.
    """)

st.divider()

# --- Sidebar or Main Form Inputs ---
st.subheader("⚙️ Input Parameters")

col1, col2 = st.columns(2)

with col1:
    rs = st.number_input("R_s (Interfacial Resistance)", value=1.0e-6, format="%.2e")
    ar = st.number_input("A_r (Aspect Ratio)", value=1.0, step=1.0, min_value=1.0)

with col2:
    vf = st.number_input("V_f (Volume Fraction)", value=5.0, step=1.0, min_value=0.0)
    kr = st.number_input("K_r (Conductivity Ratio)", value=10.0, step=1.0, min_value=1.0)

st.markdown("")

# --- Prediction Logic ---
if st.button("🚀 Predict Effective Conductivity", type="primary", use_container_width=True):
    # 1. Format raw input
    raw_input = np.array([[rs, ar, vf, kr]])
    
    # 2. Scale features
    scaled_input = X_scaler.transform(raw_input)
    
    # 3. Predict scaled output
    scaled_pred = model.predict(scaled_input).reshape(-1, 1)
    
    # 4. Inverse transform prediction to original scale
    prediction = y_scaler.inverse_transform(scaled_pred)[0][0]
    
    st.success(f"### Predicted $k_{{eff}}/k_{{(i)}}$: `{prediction:.6f}`")