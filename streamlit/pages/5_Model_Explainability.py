import streamlit as st
import pandas as pd

from utils.display_images import display_banner

# --- PAGE CONFIG ---
st.set_page_config(page_title='Model Explainability', page_icon='🧠', layout='wide')
display_banner()

st.title('Model Explainability')
st.write("""
This page summarizes how each machine learning model makes its predictions — 
visualized through **SHAP explainability plots** and **feature importance rankings**.  
All results are precomputed for performance.
""")
st.markdown('---')

# --- GLOBAL CLASSIFIER ---
st.header('1. Global Delay Classifier')
st.markdown("""
A **Random Forest classifier** categorizes congestion levels into three classes:
**0 = Low**, **1 = Medium**, and **2 = High** based on features such as average speed, delay trends, and temporal indicators.

**Evaluation results (macro averages):**
- Accuracy: 0.70  
- Precision: 0.37  
- Recall: 0.52  
- F1-Score: 0.34  

The model performs well for **low congestion** (97 % precision) and captures **high congestion** with **65 % recall**, though performance is limited for medium delays due to data imbalance.
""")


col1, col2 = st.columns([2, 1])
with col1:
    st.image("assets/shap_classifier_summary.png", caption="SHAP Summary — Global Delay Classifier", width=650)
with col2:
    st.markdown("**Top 10 Feature Importances**")
    try:
        clf_importance = pd.read_csv("assets/shap_classifier_top10.csv")
        st.dataframe(clf_importance.head(10), hide_index=True, width='stretch')
    except FileNotFoundError:
        st.warning("Feature importance file not found.")

st.markdown('---')

# --- PER-ROUTE REGRESSORS ---
st.header('2. Per-Route Delay Regressors')
st.markdown("""
Per-route **Random Forest regressors** forecast continuous travel delays using time-based, spatial, and lagged traffic features.

**Average performance:**
- R²: 0.862 
- MAE: 0.023 minutes  
- RMSE: 0.536 minutes  
""")

col1, col2 = st.columns([2, 1])
with col1:
    st.image("assets/shap_per_route_summary.png", caption="SHAP Summary — Per-Route Delay Regressors", width=650)
with col2:
    st.markdown("**Top 10 Aggregated Feature Importances**")
    try:
        per_route_importance = pd.read_csv("assets/shap_per_route_top10.csv")
        st.dataframe(per_route_importance.head(10), hide_index=True, width='stretch')
    except FileNotFoundError:
        st.warning("Feature importance file not found.")


st.markdown('---')

# --- PARKING REGRESSOR ---
st.header('3. Parking Availability Regressor')
st.markdown("""
An **XGBoost regressor** predicts parking occupancy and duration based on hourly transaction data and time-based variables.

**Performance:**
- R²: 0.95  
- MAE: 0.969   
""")

col1, col2 = st.columns([2, 1])
with col1:
    st.image("assets/shap_parking_summary.png", caption="SHAP Summary — Parking Availability Regressor", width=650)
with col2:
    st.markdown("**Top 10 Feature Importances**")
    try:
        parking_importance = pd.read_csv("assets/shap_parking_top10.csv")
        st.dataframe(parking_importance.head(10), hide_index=True, width='stretch')
    except FileNotFoundError:
        st.warning("Feature importance file not found.")

st.markdown('---')
st.caption("""
Data Source: Banff Traffic Management Project (2025)  
           
Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")

