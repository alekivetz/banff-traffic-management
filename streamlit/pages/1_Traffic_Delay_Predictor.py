import streamlit as st
import pandas as pd
from datetime import datetime

from utils.display_images import display_image, display_banner
from utils.route_info import ROUTE_DESCRIPTIONS
from utils.data_loader import fetch_routes_data, fetch_classifier, fetch_single_regressor


# --- UI Setup ---
st.set_page_config(page_title='Banff Delay Predictor', page_icon='⏱️', layout='wide')
display_banner()

# --- Load data and models ---
with st.spinner('Loading route data and models...'):
    try:
        df = fetch_routes_data(for_models=True)
        clf = fetch_classifier()
    except Exception as e:
        st.error(f'Could not load route data and models: {e}')


# --- Header ---
st.title('Banff Traffic Delay Predictor')
st.write("""
    Welcome to the **Banff Traffic Delay Predictor**, an interactive dashboard that leverages machine learning 
    to **analyze and forecast traffic delays** within the town of Banff.
""")

st.subheader('Model Overview')
st.info("""

    1. **Overall Delay Classifier**: A probabilistic model estimating the likelihood of **no delay**, **minor delay**, or **major congestion** using temporal and seasonal patterns across all routes.

    2. **Per-Route Delay Regressor**: A route-specific regression model trained on historical data, including **lag** and **rolling window** features, to predict the **expected delay duration (in minutes)** for a chosen route.
""")

st.markdown('---')
st.subheader('Select Prediction Parameters')

# --- Input Controls ---
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    route = st.selectbox('Select Route:', sorted(ROUTE_DESCRIPTIONS.keys(), key=lambda x: int(x.split()[-1])))
    date_input = st.date_input('Select Date:')
    time_input = st.time_input('Select Time:')
with col2: 
    with st.expander('View Route Descriptions', expanded=False):
        for r, desc in ROUTE_DESCRIPTIONS.items():
            st.write(f'**{r}** - {desc}')
with col3:
    image = display_image('banff_map.png')
    st.image(image, width=400)

st.markdown('---')

# --- Models ---
if st.button('Predict Delay'):
    dt = datetime.combine(date_input, time_input)

    col1, col2 = st.columns([1, 1])

    # --- Left: Classifier
    with col1:
        st.header('Overall Congestion Risk (Classifier)')

        clf_features = pd.DataFrame([{
            'route': route,
            'hour': dt.hour,
            'month': dt.month,
            'day_of_week': dt.weekday(),
            'is_weekend': int(dt.weekday() in [5, 6]),
            'season': (
                'winter' if dt.month in [12, 1, 2] else
                'spring' if dt.month in [3, 4, 5] else
                'summer' if dt.month in [6, 7, 8] else 'fall')
        }])

        probs = clf.predict_proba(clf_features)[0]
        no_delay, minor_delay, major_delay = probs

        st.subheader('Predicted Delay Probabilities')
        st.write(f'🟢 No Delay: {no_delay:.2%}')
        st.progress(no_delay)
        st.write(f'🟡 Minor Delay: {minor_delay:.2%}')
        st.progress(minor_delay)
        st.write(f'🔴 Major Delay: {major_delay:.2%}')
        st.progress(major_delay)

        st.subheader('Overall Risk Assessment')
        if major_delay > 0.6:
            st.error('🔴 **High chance of major congestion.**')
        elif minor_delay > 0.6 or major_delay > 0.3:
            st.warning('🟡 **Moderate congestion expected.**')
        else:
            st.success('🟢 **Low congestion expected.**')

    # --- Right: Regressor ---
    with col2:
        st.header('Route-Specific Delay (Regressor)')
        route_df = df[df['route'] == route].copy().sort_values('timestamp')
        route_df = route_df[route_df['timestamp'] <= dt]

        if route_df.empty:
            st.error('No available past data for this route and time.')
        else:
            closest_row = route_df.iloc[-1]
            try: 
                regressor = fetch_single_regressor(route)
            except Exception as e:
                st.error(f'Could not load model for {route}: {e}')
            else:
                feature_names = getattr(regressor, 'feature_names_in_', None)
                if feature_names is None:
                    feature_names = [c for c in closest_row.index if c not in ('timestamp', 'route')]

                X_new = pd.DataFrame([closest_row]).reindex(columns=feature_names, fill_value=0)
                pred = regressor.predict(X_new)[0]
                if pred < 0:
                    pred = 0
                st.success(f'Predicted Delay: {pred:.2f} minutes')


st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
           
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")

