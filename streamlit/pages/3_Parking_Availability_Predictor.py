import streamlit as st
import pandas as pd

from utils.display_images import display_banner
from utils.data_loader import load_parking_resources

# --- Initialization ---
with st.spinner('Loading parking resources...'):
    try:
        model, unit_encoder, features_df = load_parking_resources()
    except Exception as e:
        st.error(f'Could not load parking resources: {e}')
    
# --- UI Setup ---
st.set_page_config(page_title='Parking Availability Predictor', page_icon='⏱️', layout='wide')
display_banner()

# --- Header ---
st.title('Parking Availability Predictor - 60 Minute Forecast')
st.write("""
This interactive dashboard uses a machine learning model to **forecast parking occupancy 60 minutes ahead** 
for various lots in the town of Banff, helping visitors and planners manage congestion more effectively.
""")
st.write('Select a parking lot to view its predicted occupancy 60 minutes ahead.')  

st.markdown('---')  

# --- Select Parking Lot ---

units = sorted([u for u in features_df['unit'].unique() if pd.notna(u)])
selected_unit = st.selectbox('Select Parking Lot', units)

unit_data = features_df[features_df['unit'] == selected_unit].copy()
if unit_data.empty:
    st.error('No data available for this parking lot.')
    st.stop()

unit_data = unit_data.sort_values('ts')
latest_row = unit_data.iloc[-1]

st.markdown('---')
st.info(f'Using latest data timestamp for this lot: **{latest_row["ts"]}**')

# --- Prepare Features ---
if 'unit_encoded' not in unit_data.columns:
    unit_data['unit_encoded'] = unit_encoder.transform(unit_data['unit'])
    latest_row = unit_data.iloc[-1] 

FEATURE_COLS = [
    'occupancy', 'max_capacity',
    'lag_1', 'lag_2', 'lag_3', 'lag_4',
    'roll_mean_4', 'roll_mean_8',
    'hour', 'dayofweek', 'is_weekend', 'month',
    'unit_encoded'
]

X_current = pd.DataFrame([latest_row[FEATURE_COLS].to_dict()])

# --- Predict ---
pred_occ = int(model.predict(X_current)[0])
pred_occ = max(0, min(pred_occ, int(latest_row['max_capacity'])))

available = int(round(latest_row['max_capacity'] - pred_occ))

# --- Display ---
st.subheader(f'Selected Parking Lot: {selected_unit}')

col1, col2, col3 = st.columns(3)  
with col1:
    st.metric('Current occupancy', f'{int(latest_row["occupancy"])}/{int(latest_row["max_capacity"])}')
with col2:
    st.metric('Predicted occupancy in 60 min', f'{pred_occ}/{int(latest_row["max_capacity"])}')
with col3: 
    st.metric('Estimated free spaces in 60 min', available)

if available <= 0:
    st.error('🔴 Lot will likely be FULL in 60 minutes.')
elif available < latest_row['max_capacity'] * 0.1:
    st.warning('🟡 Lot will be almost full in 60 minutes.')
else:
    st.success('🟢 Spaces should be available in 60 minutes.')

st.caption('Forecasts generated using an XGBoost regression model trained on 15-minute occupancy intervals.')

# --- Footer ---
st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
           
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")
