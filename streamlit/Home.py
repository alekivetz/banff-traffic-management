import streamlit as st
from google.oauth2 import service_account
from utils.display_images import display_banner

def main():
    st.set_page_config(
        page_title='Banff Traffic Management',
        page_icon='🚗',
        layout='wide',
    )

    # --- Banner ---
    display_banner()

    # --- Header ---
    st.title('Alpine Analysts: Banff Traffic Management')
    st.write("""
    Welcome to the **Banff Traffic Management System**, a suite of machine learning-powered dashboards 
    developed by **Team Alpine Analysts** to analyze, visualize, and forecast traffic and parking trends 
    across the town of Banff.
    """)

    st.markdown('---')
    st.subheader('Project Overview')
    st.write("""
    This tool integrates multiple predictive and analytical components designed to support smarter mobility decisions:
             
    * **Traffic Delay Predictor** – Estimates overall congestion risk and route-specific delay durations.
    * **Traffic Analysis Dashboard** – Visualizes historical travel times, volumes, and delay patterns.
    * **Parking Availability Predictor** – Predicts parking lot occupancy 60 minutes ahead.
    * **Parking Analysis Dashboard** – Provides interactive visual analytics for parking sessions, revenue, and activity trends.
    * **Model Explainability** – Highlights which features most influenced each model’s predictions using SHAP values and importance rankings.
    * **Project Chatbot** – A retrieval-augmented assistant that answers questions about Banff traffic, parking, model performance, and project methodology in natural language.  
    * **About the Project** – Learn about data sources, modeling approach, and the Alpine Analysts team.
    """)

    st.info("""
    Use the **sidebar** to navigate between pages.  
    Each section provides insights into a different aspect of Banff’s transportation ecosystem, combining real-world data, predictive modeling, and interactive visualization.
    """)

    st.markdown('---')
    st.caption('Created by Alpine Analysts · NorQuest College MLAD · Fall 2025')

if __name__ == '__main__':
    main()
