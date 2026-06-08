import streamlit as st
import pandas as pd
import joblib
import datetime
import math
import os
import numpy as np

# --- Safely import plotly ---
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Superstore Profit Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE INIT ----------------
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ---------------- MASTER CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@700;800&family=Manrope:wght@500;600&family=Sora:wght@600;700&family=Space+Grotesk:wght@700;800&display=swap');

html { scroll-behavior: smooth; }

..hero-title { font-family: 'Space Grotesk', sans-serif !important; font-weight: 800 !important; }
h1, h2, h3, h4, h5, .title-box { font-family: 'Sora', sans-serif !important; font-weight: 700 !important; }
.result-amount, [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-weight: 800 !important; }
.micro-text { font-family: 'Manrope', sans-serif !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.1em; }
p, div.stMarkdown { font-family: 'Inter', sans-serif !important; }

[data-testid="stMetric"] { background: rgba(15, 23, 42, 0.4) !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; border-radius: 16px !important; padding: 20px !important; }

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #172554 50%, #0f172a 100%);
    background-size: 200% 200%;
    animation: gradientMove 20s ease infinite alternate;
    color: #f8fafc;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
@keyframes gradientMove { 0% { background-position: 0% 0%; } 100% { background-position: 100% 100%; } }

/* 1. MAIN HERO TITLES */
.hero-title, .hero-title span { font-family: 'Space Grotesk', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.03em !important; }
/* 2. SECTION HEADINGS */
h1:not(.hero-title), h2, h3, h4, h5, h6, .title-box, .rec-card h4 { font-family: 'Sora', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.01em !important; }
/* 3. KPI / METRIC NUMBERS */
.result-amount, [data-testid="stMetricValue"], [data-testid="stMetricValue"] div, th, td, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { font-family: 'JetBrains Mono', monospace !important; font-weight: 800 !important; font-variant-numeric: tabular-nums !important; }
/* 4. BODY CONTENT */
p:not(.micro-text):not(.footer-text), span:not(.risk-badge), div.stMarkdown, .insight-box, li[role="option"] { font-family: 'Inter', sans-serif !important; font-weight: 400 !important; line-height: 1.6 !important; }
b, strong { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
/* 5. LABELS & MICRO TEXT */
label, .micro-text, [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, .risk-badge, div[role="radiogroup"] label p, div[class*="-placeholder"], input::placeholder { font-family: 'Manrope', sans-serif !important; font-weight: 600 !important; }
/* 6. BUTTON TEXT */
div.stButton > button, div.stButton > button p { font-family: 'Sora', sans-serif !important; font-weight: 600 !important; letter-spacing: 0.5px !important; }
/* 8. FOOTER TYPOGRAPHY */
.footer-text { font-family: 'Manrope', sans-serif !important; font-weight: 500 !important; }

/* --- Layout & Glassmorphism --- */
.block-container { background: rgba(10, 15, 30, 0.75); backdrop-filter: blur(35px); -webkit-backdrop-filter: blur(35px); padding: clamp(20px, 5vw, 60px) clamp(20px, 6vw, 80px) !important; border-radius: 40px; box-shadow: 0 40px 100px -20px rgba(0, 0, 0, 0.9); max-width: 1200px; border: 1px solid rgba(255, 255, 255, 0.1); margin-top: clamp(10px, 3vw, 40px); margin-bottom: 40px; z-index: 1; }
[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
[data-testid="column"] { display: flex; flex-direction: column; }
[data-testid="column"] > div { flex: 1; display: flex; flex-direction: column; }
[data-testid="stVerticalBlockBorderWrapper"] { flex: 1 !important; background: rgba(30, 41, 59, 0.45) !important; backdrop-filter: blur(20px) !important; padding: clamp(20px, 3vw, 45px) clamp(20px, 3vw, 55px) !important; border-radius: 32px !important; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; margin-bottom: 35px !important; overflow: visible !important; transition: all 0.3s ease !important; position: relative; z-index: 2; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow: 0 25px 60px rgba(6, 182, 212, 0.2) !important; border-color: rgba(6, 182, 212, 0.4) !important; transform: translateY(-2px); }
.title-box-container { text-align: center; margin-bottom: 30px; margin-top: -15px; }
.title-box { display: inline-block; background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(6, 182, 212, 0.5); padding: 10px 30px; border-radius: 16px; font-size: clamp(16px, 2vw, 18px); color: #fff; text-transform: uppercase; letter-spacing: 1.5px; box-shadow: 0 10px 25px rgba(0,0,0,0.5), inset 0 0 15px rgba(6, 182, 212, 0.2); }
[data-testid="stNumberInput"] div[data-baseweb="base-input"], [data-testid="stTextInput"] div[data-baseweb="base-input"], [data-testid="stNumberInput"] > div > div, [data-testid="stTextInput"] > div > div { background-color: transparent !important; border: none !important; }
[data-testid="stNumberInput"] div[data-baseweb="input"], [data-testid="stTextInput"] div[data-baseweb="input"], [data-testid="stSelectbox"] div[data-baseweb="select"] { background-color: rgba(15, 23, 42, 0.8) !important; color: #f8fafc !important; border-radius: 12px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; }
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { background-color: transparent !important; color: white !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] div { background-color: transparent !important; }
div[class*="-placeholder"], input::placeholder { color: rgba(255, 255, 255, 0.4) !important; }
div[class*="-singleValue"] { color: white !important; }
[data-testid="stSelectbox"] input { pointer-events: none !important; caret-color: transparent !important; }
[data-testid="stNumberInput"] div[data-baseweb="input"]:hover, [data-testid="stTextInput"] div[data-baseweb="input"]:hover, [data-testid="stSelectbox"] div[data-baseweb="select"]:hover { transform: translateY(-3px) scale(1.02) !important; box-shadow: 0 8px 20px rgba(6, 182, 212, 0.3) !important; border-color: #06b6d4 !important; }
[data-testid="stNumberInput"] button { display: none !important; }
[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within, [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within, [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within { border-color: #06b6d4 !important; box-shadow: 0 0 15px rgba(6, 182, 212, 0.4) !important; transform: translateY(-2px) !important; }
li[role="option"] { transition: all 0.2s ease !important; border-radius: 8px !important; margin: 2px 5px !important; }
li[role="option"]:hover { background: linear-gradient(90deg, rgba(6, 182, 212, 0.2), transparent) !important; transform: scale(1.02) translateX(5px) !important; color: #06b6d4 !important; font-weight: bold !important; }
div[role="radiogroup"] { display: grid !important; grid-template-columns: repeat(4, 1fr) !important; gap: 12px !important; width: 100% !important; }
div[role="radiogroup"] label { background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.15); border-radius: 25px !important; padding: 10px 15px !important; transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important; cursor: pointer; display: flex !important; align-items: center !important; justify-content: center !important; gap: 8px !important; width: 100% !important; box-sizing: border-box !important; }
div[role="radiogroup"] label:hover { background: rgba(6, 182, 212, 0.15); border-color: #06b6d4; transform: translateY(-4px) scale(1.02); box-shadow: 0 8px 20px rgba(6, 182, 212, 0.3); }
div[role="radiogroup"] label:active { transform: scale(0.92) translateY(2px) !important; }
div[role="radiogroup"] label[data-checked="true"] { background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important; border-color: transparent !important; color: white !important; box-shadow: 0 5px 15px rgba(59, 130, 246, 0.5) !important; transform: translateY(-2px); }
div[role="radiogroup"] label > div:first-child { margin: 0 !important; }
div[role="radiogroup"] label p, div[role="radiogroup"] label div.stMarkdown { margin: 0 !important; padding: 0 !important; font-weight: 600 !important; }

[data-testid="stMetric"] { background: rgba(15, 23, 42, 0.6) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 16px !important; padding: 15px !important; box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important; transition: all 0.3s ease !important; position: relative !important; overflow: hidden !important; }
[data-testid="stMetric"]:hover { border-color: #06b6d4 !important; background: #0f172a !important; transform: scale(1.02) !important; box-shadow: 0 10px 25px rgba(6, 182, 212, 0.3) !important; }
[data-testid="stMetricLabel"] { font-size: 13px !important; color: #cbd5e1 !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { font-size: clamp(20px, 2.5vw, 26px) !important; color: #fff !important; transition: all 0.3s ease !important; }

@keyframes popUpBounce { 0% { transform: scale(0.6); opacity: 0; } 60% { transform: scale(1.02); opacity: 1; } 100% { transform: scale(1); opacity: 1; } }
@keyframes smoothFadeIn { 0% { opacity: 0; transform: translateY(20px); } 100% { opacity: 1; transform: translateY(0); } }
@keyframes pulseGlowProfit { 0% { box-shadow: 0 0 15px rgba(52, 211, 153, 0.2); } 50% { box-shadow: 0 0 40px rgba(52, 211, 153, 0.8), inset 0 0 15px rgba(52, 211, 153, 0.2); } 100% { box-shadow: 0 0 15px rgba(52, 211, 153, 0.2); } }
@keyframes pulseGlowLoss { 0% { box-shadow: 0 0 15px rgba(248, 113, 113, 0.2); } 50% { box-shadow: 0 0 40px rgba(248, 113, 113, 0.8), inset 0 0 15px rgba(248, 113, 113, 0.2); } 100% { box-shadow: 0 0 15px rgba(248, 113, 113, 0.2); } }

.profit-glow-box { background: linear-gradient(145deg, rgba(6,78,59,0.6), rgba(2,44,34,0.9)); border: 1px solid rgba(16,185,129,0.6); padding: 30px; border-radius: 20px; text-align: center; animation: popUpBounce 0.5s forwards, pulseGlowProfit 2.5s infinite alternate !important; }
.loss-glow-box { background: linear-gradient(145deg, rgba(127,29,29,0.6), rgba(69,10,10,0.9)); border: 1px solid rgba(239,68,68,0.6); padding: 30px; border-radius: 20px; text-align: center; animation: popUpBounce 0.5s forwards, pulseGlowLoss 2.5s infinite alternate !important; }
.result-amount { font-size: clamp(40px, 5vw, 65px); margin: 10px 0; }
.chart-arrival { animation: smoothFadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
div.stButton > button { background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important; color: white !important; border: none !important; border-radius: 20px !important; height: 80px !important; width: 100% !important; transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important; box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.5) !important; text-transform: uppercase; display: flex !important; align-items: center !important; justify-content: center !important; }
div.stButton > button:hover { transform: translateY(-5px) scale(1.02) !important; box-shadow: 0 20px 40px -5px rgba(6, 182, 212, 0.7) !important; background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%) !important; }
div.stButton > button:active { transform: translateY(2px) scale(0.98) !important; }
div.stButton > button p { font-size: clamp(14px, 1.3vw, 18px) !important; margin: 0 !important; }
.insight-box { background: rgba(16, 185, 129, 0.1); border-left: 5px solid #10b981; padding: 15px 20px; border-radius: 10px; font-size: 16px; margin-top: 20px; text-align: left; }
.insight-box.loss { background: rgba(239, 68, 68, 0.1); border-left: 5px solid #ef4444; }
.rec-card { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; height: 100%; transition: all 0.3s ease; }
.rec-card:hover { transform: translateY(-5px); border-color: #3b82f6; box-shadow: 0 10px 20px rgba(59, 130, 246, 0.2); }
.risk-badge { padding: 5px 15px; border-radius: 20px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; display: inline-block; }
.risk-low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
.risk-med { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
.risk-high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
.hover-glow-chart .barlayer path, .hover-glow-chart .pielayer path { transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; transform-box: fill-box !important; transform-origin: center !important; }
.hover-glow-chart .barlayer:hover path, .hover-glow-chart .pielayer:hover path { opacity: 0.3 !important; }
.hover-glow-chart .barlayer path:hover, .hover-glow-chart .pielayer path:hover { opacity: 1 !important; transform: scale(1.02) !important; filter: drop-shadow(0px 8px 15px rgba(255, 255, 255, 0.3)) !important; stroke: #ffffff !important; stroke-width: 2px !important; z-index: 999 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_components():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "superstore_profit_model.pkl")
    cols_path = os.path.join(BASE_DIR, "model_columns.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not os.path.exists(cols_path):
        raise FileNotFoundError(f"Missing columns file: {cols_path}")

    model = joblib.load(model_path)
    cols = joblib.load(cols_path)

    if not hasattr(model, "predict"):
        raise TypeError("Invalid model object loaded")

    if not isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
        raise TypeError("Invalid model columns object")

    return model, cols

try:
    model, model_columns = load_components()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Ensure 'superstore_profit_model.pkl' and 'model_columns.pkl' exist in the script directory.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div id="top-of-page"></div>
<div style="text-align:center; padding: 5px 0; margin-bottom: 30px;">
    <h1 class="hero-title" style="font-size:clamp(40px, 5vw, 60px); background:linear-gradient(to right, #22d3ee, #60a5fa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom: 8px;">
        SUPERSTORE PROFIT ANALYZER
    </h1>
    <p class="micro-text" style="color:#64748b; font-size:clamp(12px, 1.5vw, 14px); margin-top: 0px; letter-spacing: 2px; text-transform: uppercase;">
        DATA-DRIVEN BUSINESS INTELLIGENCE
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- BLOCK 1: ORDER DETAILS ----------------
with st.container(border=True):
    st.markdown('<div class="title-box-container"><div class="title-box">🛒 Order Details</div></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<p style='color:#22d3ee; font-weight:bold; font-size: 16px; margin-bottom: 10px;'>💰 Financial Settings</p>", unsafe_allow_html=True)
        
        if "sales_in" not in st.session_state:
            st.session_state.sales_in = ""
        if "discount_in" not in st.session_state:
            st.session_state.discount_in = ""

        def format_sales():
            val_str = str(st.session_state.sales_in).replace(',', '').strip()
            if val_str:
                try: 
                    val = float(val_str)
                    if not (math.isinf(val) or math.isnan(val) or val < 0):
                        st.session_state.sales_in = f"{val:.2f}"
                except ValueError: pass

        def format_discount():
            val_str = str(st.session_state.discount_in).replace(',', '').strip()
            if val_str:
                try: 
                    val = float(val_str)
                    if not (math.isinf(val) or math.isnan(val) or val < 0 or val > 0.99):
                        st.session_state.discount_in = f"{val:.2f}"
                except ValueError: pass

        sales_input = st.text_input("Sales Amount ($)", key="sales_in", placeholder="Eg. 150.00", on_change=format_sales)
        # FIX 3: Removed explicit `value=""` argument to prevent State Glitches
        quantity_input = st.text_input("Quantity", key="qty_in", placeholder="Eg. 2")
        discount_input = st.text_input("Discount (0.0 - 0.99)", key="discount_in", placeholder="Eg. 0.10", on_change=format_discount)
        
        sales, quantity, discount = None, None, None
        
        # Validated Input Logic
        sales_clean = str(sales_input).replace(',', '').strip() if sales_input else ""
        if sales_clean:
            try:
                val = float(sales_clean)
                if math.isinf(val) or math.isnan(val) or val < 0:
                    st.error("⚠️ Invalid input! Please enter a positive numeric Sales Amount.")
                else:
                    sales = val
            except ValueError:
                st.error("⚠️ Invalid input! Please enter a numeric Sales Amount.")
                
        qty_clean = str(quantity_input).replace(',', '').strip() if quantity_input else ""
        if qty_clean:
            if qty_clean.isdigit():
                val = int(qty_clean)
                if val < 1:
                    st.error("⚠️ Quantity must be at least 1.")
                else:
                    quantity = val
            else:
                st.error("⚠️ Invalid input! Please enter a positive whole number for Quantity.")
                
        disc_clean = str(discount_input).replace(',', '').strip() if discount_input else ""
        if disc_clean:
            try:
                d = float(disc_clean)
                if math.isinf(d) or math.isnan(d):
                    st.error("⚠️ Invalid input! Please enter a numeric Discount.")
                elif 0.0 <= d <= 0.99:
                    discount = d
                else:
                    st.error("⚠️ Discount must be between 0.0 and 0.99.")
            except ValueError:
                st.error("⚠️ Invalid input! Please enter a numeric Discount.")

    with col2:
        st.markdown("<p style='color:#22d3ee; font-weight:bold; font-size: 16px; margin-bottom: 10px;'>📦 Product & Market</p>", unsafe_allow_html=True)
        category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"], key="cat_in", index=None, placeholder="Select Category")
        segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"], key="seg_in", index=None, placeholder="Select Segment")
        region = st.selectbox("Region", ["South", "West", "Central", "East"], key="reg_in", index=None, placeholder="Select Region")

# ---------------- BLOCK 2: TIMING & CONFIG ----------------
col_t1, col_t2 = st.columns(2, gap="large")

with col_t1:
    with st.container(border=True):
        st.markdown('<div class="title-box-container"><div class="title-box">⏱️ Order Timing</div></div>', unsafe_allow_html=True)
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        
        selected_month_name = st.radio("Month", month_names, key="month_in", horizontal=True, label_visibility="collapsed", index=None)
        month = {name:i+1 for i,name in enumerate(month_names)}[selected_month_name] if selected_month_name else None
        
        st.markdown("<br>", unsafe_allow_html=True)

        current_yr = datetime.datetime.now().year
        # FIX 3: Removed explicit `value=""` argument to prevent State Glitches
        year_input = st.text_input(f"📅 Enter Order Year ({current_yr - 5} - {current_yr + 5})", key="year_in", placeholder=str(current_yr))
        
        year_clean = str(year_input).replace(',', '').strip() if year_input else ""
        if year_clean:
            if year_clean.isdigit():
                y = int(year_clean)
                if (current_yr - 5) <= y <= (current_yr + 5):
                    year = y
                else:
                    st.error(f"⚠️ Please enter a year between {current_yr - 5} and {current_yr + 5}.")
                    year = None
            else:
                st.error("⚠️ Invalid input! Please enter a numeric year.")
                year = None
        else:
            year = None

with col_t2:
    with st.container(border=True):
        st.markdown('<div class="title-box-container"><div class="title-box">🔍 Configuration</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Sales", f"${sales:,.0f}" if sales is not None else "N/A")
        c2.metric("📦 Quantity", quantity if quantity is not None else "N/A")
        c3.metric("🏷️ Discount", f"{discount*100:.0f}%" if discount is not None else "N/A")
        st.markdown("<br>", unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        c4.metric("🏢 Category", category if category else "N/A")
        c5.metric("🌍 Region", region if region else "N/A")
        c6.metric("📅 Date", f"{selected_month_name if selected_month_name else 'MM'} '{str(year)[-2:] if year else 'YY'}")

# ---------------- PREDICTION LOGIC & STATE MANAGEMENT ----------------
st.markdown('<br>', unsafe_allow_html=True)

if st.button("🚀 ANALYZE & PREDICT PROFITABILITY", use_container_width=True):
    if sales is None or quantity is None or discount is None or not category or not segment or not region or not selected_month_name or year is None:
        st.error("⚠️ Please fill in ALL Order Details, Month, and Year before predicting!")
        st.session_state.show_results = False
    else:
        st.session_state.show_results = True
        
        base_dict = {
            "Sales": sales, "Quantity": quantity, "Discount": discount,
            "Month": month, "Year": year, "Segment": segment,
            "Region": region, "Category": category
        }
        st.session_state.inputs = {k: [v] for k, v in base_dict.items()}
        
        scenarios = [base_dict.copy()]
        
        # 1. Insight Recovery Scenarios
        recovery_discounts = []
        if discount > 0:
            cur_d = round(discount - 0.05, 2)
            while cur_d >= 0.0:
                recovery_discounts.append(cur_d)
                cur_d = round(cur_d - 0.05, 2)
            if 0.0 not in recovery_discounts:
                recovery_discounts.append(0.0)

        recovery_start_idx = len(scenarios)
        for d in recovery_discounts:
            d_dict = base_dict.copy()
            d_dict['Discount'] = d
            scenarios.append(d_dict)
        
        # 2. Stability Scenarios (-0.05, 0, +0.05)
        stab_start_idx = len(scenarios)
        for d_adj in [-0.05, 0, 0.05]:
            s_dict = base_dict.copy()
            s_dict['Discount'] = max(0, min(0.99, discount + d_adj))
            scenarios.append(s_dict)
        stab_end_idx = len(scenarios)
        
        # 3. Improvement Scenarios
        imp_start_idx = len(scenarios)
        i_disc = base_dict.copy(); i_disc['Discount'] = max(0, discount - 0.05); scenarios.append(i_disc)
        i_sales = base_dict.copy(); i_sales['Sales'] = sales * 1.10; scenarios.append(i_sales)
        i_qty = base_dict.copy()
        i_qty['Quantity'] = quantity + 1
        i_qty['Sales'] = sales * ((quantity + 1) / quantity) if quantity > 0 else sales
        scenarios.append(i_qty)
        imp_end_idx = len(scenarios)
        
        # 4. Discount Analysis Curve
        curve_disc = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        curve_start_idx = len(scenarios)
        for d in curve_disc:
            c_dict = base_dict.copy(); c_dict['Discount'] = d; scenarios.append(c_dict)
        curve_end_idx = len(scenarios)
        
        # 5. Category Profitability Scenarios
        cats = ['Technology', 'Office Supplies', 'Furniture']
        cat_start_idx = len(scenarios)
        for c in cats:
            cat_dict = base_dict.copy(); cat_dict['Category'] = c; scenarios.append(cat_dict)
        cat_end_idx = len(scenarios)
        
        # Feature Engineering Matrix
        batch_df = pd.DataFrame(scenarios)
        encoded_matrix = pd.DataFrame(0.0, index=np.arange(len(batch_df)), columns=model_columns)
        
        # FIX 1: Prevent Feature Skew by moving 'Month' and 'Year' back to Numeric processing
        numeric_cols = ['Sales', 'Quantity', 'Discount', 'Month', 'Year']
        cat_cols = ['Segment', 'Region', 'Category']
        
        for idx, row in batch_df.iterrows():
            for num_col in numeric_cols:
                if num_col in model_columns:
                    encoded_matrix.at[idx, num_col] = row[num_col]
                    
            for c_col in cat_cols:
                feature_name = f"{c_col}_{row[c_col]}"
                if feature_name in model_columns:
                    encoded_matrix.at[idx, feature_name] = 1.0 


        if encoded_matrix.isnull().values.any():
            st.error("Feature matrix contains NaN values.")
            st.stop()

        if np.isinf(encoded_matrix.values).any():
            st.error("Feature matrix contains Infinite values.")
            st.stop()        
        
        # O(1) Single Inference Call
        all_preds = model.predict(encoded_matrix)
        
        prediction = float(all_preds[0])
        if np.isnan(prediction):
            st.error("Model returned NaN prediction.")
            st.stop()

        if np.isinf(prediction):
            st.error("Model returned Infinite prediction.")
            st.stop()

        st.session_state.prediction = prediction

        # Extract Results
        st.session_state.prediction = float(all_preds[0])
        st.session_state.profit_margin = (st.session_state.prediction / sales) * 100 if sales >= 0.01 else 0.0
        
        # FIX 4: Removed branching logic bloat. Cost = Revenue - Profit is unified for both + and - predictions.
        st.session_state.cost = max(0.0, sales - st.session_state.prediction)
        
        # Prevent division-by-zero risk logically, though UI is capped at 0.99
        st.session_state.discount_amount = (sales / (1 - discount) - sales) if discount < 0.999 else 0.0

        st.session_state.data_insight = ""
        if st.session_state.prediction < 0:
            for idx, d in enumerate(recovery_discounts):
                if all_preds[recovery_start_idx + idx] > 0:
                    st.session_state.data_insight = f"💡 <b>Strategic Recommendation:</b> Reducing the discount to <b>{d*100:.0f}%</b> recovers the margin, yielding an estimated profit of <b>${all_preds[recovery_start_idx + idx]:,.2f}</b>."
                    break
            if not st.session_state.data_insight:
                st.session_state.data_insight = "⚠️ <b>Strategic Recommendation:</b> This product category is historically unprofitable at this volume/region. Consider rejecting the order or increasing base price."
        else:
            st.session_state.data_insight = f"💡 <b>Data Insight:</b> Strong configuration with a healthy <b>{st.session_state.profit_margin:.1f}%</b> margin. Safe to process this order."
            
        stab_preds = all_preds[stab_start_idx:stab_end_idx]
        mean_pred = sum(stab_preds) / len(stab_preds)
        variation = max(stab_preds) - min(stab_preds)
        
        stability_score = 100 if abs(mean_pred) < 1 else max(0, 100 - (variation / (abs(mean_pred) + 1)) * 100)
        st.session_state.stability_score = int(max(0, min(100, stability_score)))
        
        margin_score = max(0, min(100, st.session_state.profit_margin * 2))
        volatility_penalty = min(50, (variation / (abs(st.session_state.prediction) + 1)) * 100)
        health_raw = margin_score - volatility_penalty + (50 if st.session_state.prediction > 0 else 0)
        st.session_state.health_score = int(max(0, min(100, health_raw)))
        
        imp_preds = all_preds[imp_start_idx:imp_end_idx]
        st.session_state.scenario_improvements = {
            'discount': float(imp_preds[0]) - st.session_state.prediction,
            'sales': float(imp_preds[1]) - st.session_state.prediction,
            'qty': float(imp_preds[2]) - st.session_state.prediction
        }
        
        st.session_state.curve_preds = [float(p) for p in all_preds[curve_start_idx:curve_end_idx]]
        st.session_state.cat_preds = [float(p) for p in all_preds[cat_start_idx:cat_end_idx]]

        st.session_state.prediction_history.append({
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "Sales": f"${sales:,.2f}",
            "Qty": quantity,
            "Discount": f"{discount*100:.0f}%",
            "Category": category,
            "Segment": segment,
            "Region": region,
            "Month": selected_month_name,
            "Year": year,
            "Prediction": f"${st.session_state.prediction:,.2f}",
            "Stab.": f"{st.session_state.stability_score}%"
        })
        if len(st.session_state.prediction_history) > 10:
            st.session_state.prediction_history.pop(0)

# ---------------- RESULTS DASHBOARD ----------------
if st.session_state.show_results:
    pred = st.session_state.prediction
    margin = st.session_state.profit_margin
    cost = st.session_state.cost
    disc_amt = st.session_state.discount_amount
    insight = st.session_state.data_insight
    stability = st.session_state.stability_score
    health = st.session_state.health_score

    st.markdown("<div id='result-box' style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="title-box-container"><div class="title-box">📊 Profitability Analysis Report</div></div>', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1.2, 1], gap="large")
        
        with res_col1:
            if pred > 0:
                st.markdown(f"""
                <div class="profit-glow-box">
                    <h3 style="color: #6ee7b7; margin: 0; font-weight: 800;">🎉 ESTIMATED NET PROFIT</h3>
                    <h1 class="result-amount" style="color: #34d399; text-shadow: 0 0 20px rgba(52,211,153,0.5);">+${pred:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="loss-glow-box">
                    <h3 style="color: #fca5a5; margin: 0; font-weight: 800;">⚠️ ESTIMATED NET LOSS</h3>
                    <h1 class="result-amount" style="color: #f87171; text-shadow: 0 0 20px rgba(248,113,113,0.5);">-${abs(pred):,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<div class='insight-box loss'>{insight}</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown(f"**🛡️ Statistical Stability Index: {stability}%**")
                st.progress(stability/100)
            with hc2:
                st.markdown(f"**❤️ Financial Health Index: {health}/100**")
                st.progress(health/100)
            
            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("📉 Margin", f"{margin:.1f}%")
            k2.metric("💸 Est. Cost", f"${cost:,.0f}")
            k3.metric("✂️ Discount", f"${disc_amt:,.0f}")

        with res_col2:
            if PLOTLY_AVAILABLE:
                labels = ['Cost', 'Net Profit'] if pred > 0 else ['Revenue', 'Shortfall (Loss)']
                values = [cost, pred] if pred > 0 else [st.session_state.inputs['Sales'][0], abs(pred)]
                pie_colors = ['#334155', '#10b981'] if pred > 0 else ['#334155', '#ef4444']
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=.5, 
                    marker_colors=pie_colors, textinfo='percent+label',
                    textfont=dict(color='white', size=13), hoverinfo='label+value'
                )])
                
                fig.update_traces(marker=dict(line=dict(color='#0f172a', width=3)), pull=[0.05, 0])
                fig.update_layout(
                    title_text="Financial Breakdown", 
                    title_font=dict(size=18, color="white", family="Sora"),
                    font=dict(family="Inter", size=13),
                    title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=40, b=10, l=10, r=10), showlegend=False
                )
                
                st.markdown('<div class="chart-arrival">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
                
                gauge_color = "#10b981" if pred > 0 else "#ef4444"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=pred, 
                    title={'text': "Profitability Gauge", 'font': {'color': 'white', 'family': 'Sora'}},
                    number={'font': {'color': gauge_color, 'family': 'JetBrains Mono'}, 'prefix': "$" if pred > 0 else "-$"},
                    gauge={ 'axis': {'range': [min(-500, pred-100), max(1000, pred+200)], 'tickcolor': "white", 'tickfont': {'family': 'JetBrains Mono'}}, 'bar': {'color': gauge_color} }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.warning("📊 Install 'plotly' to view the visual breakdown charts.")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="title-box-container"><div class="title-box">🚀 Business Intelligence Insights</div></div>', unsafe_allow_html=True)
            
            st.markdown("### 💡 Strategic Recommendations")
            rec1, rec2, rec3 = st.columns(3)
            with rec1:
                st.markdown(f"<div class='rec-card'><h4 style='margin-top:0;'>📉 Reduce Discount</h4><p>Lowering discount by 5% improves margins.</p><b style='color:#34d399;'>Est. Impact: +${max(0, st.session_state.scenario_improvements['discount']):,.2f}</b></div>", unsafe_allow_html=True)
            with rec2:
                st.markdown(f"<div class='rec-card'><h4 style='margin-top:0;'>📈 Increase Sales</h4><p>Upselling optimizes operational costs.</p><b style='color:#34d399;'>Est. Impact: +${max(0, st.session_state.scenario_improvements['sales']):,.2f}</b></div>", unsafe_allow_html=True)
            with rec3:
                st.markdown(f"<div class='rec-card'><h4 style='margin-top:0;'>🎯 Optimize Quantity</h4><p>Bundling items increases profitability.</p><b style='color:#34d399;'>Est. Impact: +${max(0, st.session_state.scenario_improvements['qty']):,.2f}</b></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("### 🛡️ Risk Assessment")
            risk_lvl = "High" if pred < 0 else "Medium" if margin < 10 else "Low"
            risk_cls = "risk-high" if risk_lvl == "High" else "risk-med" if risk_lvl == "Medium" else "risk-low"
            st.markdown(f"**Current Risk Level:** <span class='risk-badge {risk_cls}'>{risk_lvl} Risk</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            with risk_col1:
                st.markdown(f"<div class='rec-card'><h4 style='margin-top:0;'>📦 Volume Status</h4><p>Sales volume is <b>{'Adequate' if (st.session_state.inputs['Sales'][0]) > 100 else 'Low'}</b>.</p></div>", unsafe_allow_html=True)
            with risk_col2:
                st.markdown(f"<div class='rec-card'><h4 style='margin-top:0;'>🏷️ Discount Check</h4><p>Discount is <b>{'Dangerously High' if (st.session_state.inputs['Discount'][0]) > 0.15 else 'Acceptable'}</b>.</p></div>", unsafe_allow_html=True)
            with risk_col3:
                st.markdown(f"<div class='rec-card'><h4 style='margin-top:0;'>🌍 Region Outlook</h4><p>Market outlook is <b>{'Favorable' if pred > 0 else 'Challenging'}</b>.</p></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- DISCOUNT ANALYSIS & CHARTS ---
            if PLOTLY_AVAILABLE:
                st.markdown("### 📉 Discount Analysis")
                st.markdown("**Impact of Discount on Projected Profit**")
                
                disc_range = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
                profit_data = st.session_state.curve_preds
                
                df_impact = pd.DataFrame({'Discount (%)': [d*100 for d in disc_range], 'Profit ($)': profit_data})
                
                fig_line = px.line(df_impact, x='Discount (%)', y='Profit ($)', markers=False, template='plotly_dark')
                fig_line.update_traces(line_shape='spline', line=dict(width=4, color='#2dd4bf'))
                
                current_discount_val = (st.session_state.inputs['Discount'][0]) * 100
                fig_line.add_scatter(x=[current_discount_val], y=[pred], mode='markers', 
                                     marker=dict(color='#06b6d4', size=16, symbol='star', line=dict(color='white', width=2)),
                                     name="Current Order")
                
                # Dynamic boundary allocation preventing clipped points
                min_y_val = min(profit_data + [pred])
                max_y_val = max(profit_data + [pred])
                padding = abs(max_y_val - min_y_val) * 0.2 + 5
                maxallowed_x = max(32, current_discount_val + 5)
                
                fig_line.update_layout(
                    height=350, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    margin=dict(t=20, b=20), 
                    hovermode="x unified",
                    font=dict(family="Inter"),
                    xaxis=dict(showspikes=False, showgrid=True, gridcolor='rgba(255,255,255,0.05)', minallowed=-2, maxallowed=maxallowed_x, tickfont=dict(family="JetBrains Mono")),
                    yaxis=dict(showspikes=False, showgrid=True, gridcolor='rgba(255,255,255,0.05)', minallowed=min_y_val - padding, maxallowed=max_y_val + padding, tickfont=dict(family="JetBrains Mono")),
                    showlegend=False
                )
                
                st.markdown('<div class="chart-arrival">', unsafe_allow_html=True)
                st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': True})
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                chart_col1, chart_col2 = st.columns(2, gap="large")
                
                with chart_col1:
                    with st.container(border=True):
                        st.markdown("<h5 style='text-align:center; color:#cbd5e1; margin-bottom:0;'>📊 Predicted Category Comparison</h5>", unsafe_allow_html=True)
                        
                        cats = ['Technology', 'Office Supplies', 'Furniture']
                        scores = [round(float(p), 2) for p in st.session_state.cat_preds]
                            
                        cat_data = pd.DataFrame({'Category': cats, 'Profit': scores})
                        cat_colors = ['#60a5fa', '#3b82f6', '#1d4ed8']
                        
                        fig_bar = go.Figure(go.Bar(
                            x=cat_data['Category'], y=cat_data['Profit'], text=cat_data['Profit'],
                            marker=dict(color=cat_colors, line=dict(color='rgba(255,255,255,0.5)', width=2)),
                            textposition='inside', textfont=dict(size=18, color='white', family='JetBrains Mono')
                        ))
                        
                        fig_bar.update_layout(
                            template='plotly_dark',
                            height=250, 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            margin=dict(t=20, b=10), 
                            xaxis_title="", yaxis_title="Predicted Profit ($)", 
                            showlegend=False, hovermode="closest",
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                        )
                        
                        st.markdown('<div class="chart-arrival hover-glow-chart">', unsafe_allow_html=True)
                        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
                        st.markdown('</div>', unsafe_allow_html=True)

                with chart_col2:
                    with st.container(border=True):
                        st.markdown("<h5 style='text-align:center; color:#cbd5e1; margin-bottom:0;'>🎯 Scenario Success Ratio</h5>", unsafe_allow_html=True)
                        scenario_vals = [1 if v > 0 else 0 for v in [pred, pred + st.session_state.scenario_improvements['discount'], pred + st.session_state.scenario_improvements['sales'], pred + st.session_state.scenario_improvements['qty']]]
                        positive = sum(scenario_vals)
                        negative = len(scenario_vals)-positive
                        
                        # FIX 2: Enforced rigid discrete color map to prevent Plotly slice swapping bugs
                        scenario_data = pd.DataFrame({
                            'Scenario Type': ['Positive Scenarios', 'Negative Scenarios'],
                            'Count': [positive, negative]
                        })
                        prob_fig = px.pie(
                            scenario_data, 
                            values='Count', 
                            names='Scenario Type',
                            color='Scenario Type',
                            color_discrete_map={'Positive Scenarios': '#10b981', 'Negative Scenarios': '#ef4444'},
                            hole=0.75, 
                            template='plotly_dark'
                        )
                        
                        prob_fig.update_layout(
                            height=250, 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            margin=dict(t=10, b=10), 
                            showlegend=True,
                            font=dict(family="Inter", size=13),
                            legend=dict(font=dict(family="Manrope", size=12))
                        )
                        
                        st.markdown('<div class="chart-arrival hover-glow-chart">', unsafe_allow_html=True)
                        st.plotly_chart(prob_fig, use_container_width=True, config={'displayModeBar': False})
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📊 Install 'plotly' to view Discount and Category comparison charts.")

        st.markdown("<br>", unsafe_allow_html=True)            
        with st.container(border=True):
            st.markdown('<div class="title-box-container"><div class="title-box">🕒 Recent Prediction History</div></div>', unsafe_allow_html=True)
            if len(st.session_state.prediction_history) > 0:
                hist_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run an analysis to see your prediction history here.")

# ---------------- FOOTER ----------------
current_yr = datetime.datetime.now().year
st.markdown(f"""
<div class="footer-text" style="text-align: center; margin-top: 50px; margin-bottom: -10px; color: #475569; font-size: 12px;">
    &copy; © {current_yr} Superstore Analytics • Business Intelligence Platform • Version 2.1
</div>
""", unsafe_allow_html=True)