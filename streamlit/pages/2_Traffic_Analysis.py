import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from utils.display_images import display_banner
from utils.data_loader import fetch_routes_data

# --- Page Config ---
st.set_page_config(page_title='Banff Traffic Analysis', page_icon='📈', layout='wide')

# --- Header ---
display_banner()

# --- Load data ---
with st.spinner('Loading route data...'):
    try:
        df = fetch_routes_data()
    except Exception as e:
        st.error(f'Could not load route data: {e}')

st.title('Banff Traffic Congestion Analysis')
st.write('This interactive dashboard visualizes historical traffic data to uncover delay patterns, peak-hour behavior, and route performance within the town of Banff.')
st.markdown('---')

filters, visuals = st.columns([1, 5])

# --- Left Controls: Parameters & Filters ---
with filters:
    st.subheader('Dashboard Filters')
    
    st.markdown('**Select Routes**')
    available_routes = ['Route 1','Route 2','Route 3','Route 4','Route 5','Route 6',
                        'Route 7','Route 8','Route 10','Route 11','Route 12','Route 13']
    selected_routes = st.multiselect('Select Routes', options=available_routes,
                                     default=available_routes[:1],
                                     label_visibility='collapsed')
    
    st.markdown('**Select Date Range**')
    min_date = datetime(2024, 12, 1).date()
    max_date = datetime(2025, 9, 7).date()
    selected_dates = st.date_input('Select Date Range', value=[min_date, max_date],
                                    min_value=min_date, max_value=max_date,
                                    label_visibility='collapsed')

    st.markdown('**Select Hour Range**')
    selected_hours = st.slider("Select Hour Range", 0, 23, (0, 24))

    st.markdown('**Select Days of Week**')
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    selected_days = st.multiselect("Select Days of Week", days, default=days)

    if st.button('Reset All'):
        st.rerun()


# --- Normalize date selection safely ---
if isinstance(selected_dates, (list, tuple)):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date, end_date = min_date, max_date

# --- Guard against partial selection (only start date clicked) ---
if len(selected_dates) < 2:
    st.warning('Please select both a start and end date to view filtered data.')
    st.stop()
    
# --- Apply Filters ---
filtered_df = df[
    (df['timestamp'].dt.date >= selected_dates[0]) &
    (df['timestamp'].dt.date <= selected_dates[1]) &
    (df['route'].isin(selected_routes)) &
    (df['hour'] >= selected_hours[0]) &
    (df['hour'] <= selected_hours[1]) &
    (df['day_of_week'].isin(selected_days))
]
with visuals:
    v1, v2 = st.columns(2)
    v3, v4 = st.columns(2)
    with v1: 
        # --- Daily Delay Trends ---
        fig1 = px.line(filtered_df.groupby(['route', pd.Grouper(key='timestamp', freq='D')])['actual_delay'].mean().reset_index(),
                    x='timestamp', y='actual_delay', color='route',
                    title='Average Daily Delay by Route',
                    labels={'timestamp': 'Date', 'actual_delay': 'Average Delay (minutes)'})
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, width='stretch')
        st.caption('Shows how daily delays change over time for each route. Sharp increases highlight peak congestion periods or traffic incidents.')

    with v2:
    # --- Delay Distribution by Day ---
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        filtered_df.loc[:, 'day_of_week'] = pd.Categorical(filtered_df['day_of_week'], categories=weekday_order, ordered=True)
        fig2 = px.box(filtered_df, x='day_of_week', y='actual_delay', color='day_of_week',
                    title='Delay Distribution by Day of Week',
                    category_orders={'day_of_week': weekday_order},
                    labels={'day_of_week': 'Day of Week', 'actual_delay': 'Delay (minutes)'})
        fig2.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig2, width='stretch')
        st.caption('Compares delays across the week. Weekends typically show higher variability, reflecting heavier visitor traffic.')

    # st.markdown('---')

    with v3:
    # --- Speed vs Delay ---
        sample_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
        fig3 = px.scatter(sample_df, x='speed_kmh', y='actual_delay', color='route',
                        title='Speed vs. Delay by Route', trendline='lowess',
                        labels={'speed_kmh': 'speed_kmh', 'actual_delay': 'Delay (minutes)'},
                        hover_data=['timestamp'])
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, width='stretch')
        st.caption('Illustrates the relationship between traffic speed and delay. Lower speeds usually correspond with longer delays, especially during busy times.')

    # --- Average Speed by Route and Hour ---
    with v4:
        heatmap_data = filtered_df.pivot_table(index='route', columns='hour', values='speed_kmh', aggfunc='mean')
        fig4 = px.imshow(heatmap_data, labels=dict(x='Hour of Day', y='Route', color='speed_kmh'),
                        x=heatmap_data.columns, y=heatmap_data.index, aspect='auto',
                        color_continuous_scale='RdYlGn_r', title='Average Speed by Route and Hour of Day')
        fig4.update_xaxes(side='bottom')
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, width='stretch')
        st.caption('Shows how traffic speed changes through the day for each route. Red areas indicate faster movement, while green areas mark slower speeds that appear during mid-day congestion.')

# --- Footer ---
st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
           
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")
