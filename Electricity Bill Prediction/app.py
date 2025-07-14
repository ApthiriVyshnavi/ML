
import streamlit as st
import numpy as np
import pickle

# ========== PAGE CONFIGURATION ==========
st.set_page_config(page_title="⚡ Electricity Consumption Predictor", layout="wide")

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #e0f7fa, #ffffff);
            font-family: 'Segoe UI', sans-serif;
        }

        h2 {
            text-align: center;
            color: #0D47A1;
            font-size: 42px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }

        .stSlider > div > div > div > div {
            background: #0D47A1;
        }

        div[data-baseweb="slider"] span {
            background-color: #1565C0 !important;
        }

        .stButton > button {
            background-color: #0D47A1;
            color: white;
            border: none;
            padding: 0.6em 1.2em;
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #1976D2;
        }

        .result-box {
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
            margin-top: 20px;
        }

        .section-header {
            font-size: 22px;
            font-weight: bold;
            color: #01579B;
            margin-top: 40px;
            margin-bottom: 10px;
        }

        hr {
            border: 1px solid #B0BEC5;
            margin-top: 10px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD PRE-TRAINED OBJECTS ==========
le_city = pickle.load(open('le_city.pkl', 'rb'))
le_company = pickle.load(open('le_company.pkl', 'rb'))
scalar = pickle.load(open('scalar.pkl', 'rb'))
lin_model = pickle.load(open('lin.pkl', 'rb'))

# ========== TITLE ==========
st.markdown("<h2>⚡ Electricity Consumption Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Use the sliders and dropdowns below to predict your monthly electricity consumption.</p>", unsafe_allow_html=True)

# ========== SECTION: APPLIANCES ==========
st.markdown("<div class='section-header'>🏠 Appliance Usage</div>", unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        fan = st.slider("🌀 Number of Fans", 0, 100, 13)
        ac = st.slider("❄️ Number of Air Conditioners", 0, 20, 2)

    with col2:
        refrigerator = st.slider("🧊 Number of Refrigerators", 0, 50, 22)
        tv = st.slider("📺 Number of Televisions", 0, 50, 21)

    with col3:
        monitor = st.slider("🖥️ Number of Monitors", 0, 20, 1)

st.markdown("<hr>", unsafe_allow_html=True)

# ========== SECTION: USAGE & LOCATION ==========
st.markdown("<div class='section-header'>🌐 Usage Details</div>", unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        month = st.slider("📅 Month (1 = Jan, 12 = Dec)", 1, 12, 4)

    with col2:
        city = st.selectbox("🏙️ Select City", le_city.classes_)

    with col3:
        company = st.selectbox("🏢 Select Company", le_company.classes_)

st.markdown("<hr>", unsafe_allow_html=True)

# ========== SECTION: USAGE HOURS & TARIFF ==========
st.markdown("<div class='section-header'>⏱️ Power Usage</div>", unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        monthly_hours = st.slider("📈 Monthly Usage Hours", 0, 1000, 546)

    with col2:
        tariff = st.slider("💰 Tariff Rate (₹/unit)", 0.0, 20.0, step=0.1, value=8.4)

# ========== PREDICTION ==========
if st.button("🔍 Predict Consumption"):
    data = [fan, refrigerator, ac, tv, monitor, month, city, company, monthly_hours, tariff]
    data[6] = int(le_city.transform([data[6]]))             # Encode city
    data[7] = int(le_company.transform([data[7]]))          # Encode company
    data[8] = scalar.transform([[data[8]]])[0][0]           # Scale MonthlyHours
    data = np.array(data).reshape(1, -1)

    prediction = lin_model.predict(data)[0]

    st.markdown(
        f"""
        <div class="result-box">
            🔌 Predicted Monthly Consumption:  ₹ <span style='font-size:34px;'>{prediction:.2f} units</span>
        </div>
        """,
        unsafe_allow_html=True
    )
