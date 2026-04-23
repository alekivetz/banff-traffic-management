import streamlit as st
import pandas as pd
import plotly.express as px

from utils.display_images import display_banner
from utils.data_loader import fetch_parking_vis_chatbot

# Helper function to cache pre-aggregated summaries
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December']

@st.cache_data(ttl=3600)
def summarize_parking(df: pd.DataFrame):
    """Return pre-aggregated summaries to speed up Plotly charts."""
    # Normalize month names before grouping
    df['month'] = (
        df['month']
        .astype(str)
        .str.strip()
        .str.title()                     
        .replace({
            'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
            'Jun': 'June', 'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
            'Sept': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
        })
    )

    # Aggregations
    monthly_sessions = df.groupby('month').size().reset_index(name='sessions')
    dow_sessions = df.groupby('day_of_week').size().reset_index(name='sessions')
    monthly_revenue = df.groupby('month')['Amount'].sum().reset_index()

    # Convert to ordered categories
    monthly_sessions['month'] = pd.Categorical(monthly_sessions['month'], categories=month_order, ordered=True)
    monthly_sessions = monthly_sessions.sort_values('month')

    monthly_revenue['month'] = pd.Categorical(monthly_revenue['month'], categories=month_order, ordered=True)
    monthly_revenue = monthly_revenue.sort_values('month')

    dow_sessions['day_of_week'] = pd.Categorical(dow_sessions['day_of_week'], categories=weekday_order, ordered=True)
    dow_sessions = dow_sessions.sort_values('day_of_week')

    # Heatmap
    pivot_df = df[df['duration'] != 'NO'].copy()
    pivot_df['duration'] = pd.to_numeric(pivot_df['duration'], errors='coerce')
    heat_df = (pivot_df.groupby(['day_of_week', 'hour']).size().reset_index(name='sessions'))
    heat_df['day_of_week'] = pd.Categorical(heat_df['day_of_week'], categories=weekday_order, ordered=True)
    heat_df = heat_df.sort_values('day_of_week')

    return monthly_sessions, dow_sessions, monthly_revenue, heat_df


# --- UI ---
st.set_page_config(page_title='Banff Parking Analysis', page_icon='🚗', layout='wide')
display_banner()

# --- Initialization ---
with st.spinner('Loading parking data...'): 
    try:
        df = fetch_parking_vis_chatbot()
        monthly_df, dow_df, rev_month_df, heat_df = summarize_parking(df)
    except Exception as e:
        st.error(f'Could not load parking data: {e}')

# --- Header ---
st.title('Banff Parking Analysis')
st.write("""
The **Banff Parking Dashboard** provides an interactive overview of parking activity across the town of Banff.  
Explore key trends in **sessions, revenue, duration, and occupancy** to understand visitor behaviour and identify 
peak usage periods throughout the year.
""")

st.markdown('---')

filters, visuals = st.columns([1, 5])
with filters:
    # --- Filters ---
    st.subheader('Filter Options')

    payment_filter = st.multiselect(
        'Payment Type',
        options=df['Payment type'].unique(),
        default=df['Payment type'].unique()
    )

    unit_filter = st.multiselect(
        'Parking Lot',
        options=df['Unit'].unique(),
        default=df['Unit'].unique()
    )

    month_filter = st.multiselect('Month',
        options=df['month'].unique(),
        default=df['month'].unique()
    )

    unit_search = st.text_input('Search Unit (optional)', placeholder='e.g., Bear Street')
    type_search = st.text_input('Search Fee or Type', placeholder='e.g., Paystation or Event')

# --- Apply Filters ---
df_filtered = df[
    (df['Payment type'].isin(payment_filter)) &
    (df['Unit'].isin(unit_filter)) &
    (df['month'].isin(month_filter))
]

if unit_search:
    df_filtered = df_filtered[df_filtered['Unit'].str.contains(unit_search, case=False, na=False)]

if type_search:
    df_filtered = df_filtered[
        df_filtered['Fee'].str.contains(type_search, case=False, na=False) |
        df_filtered['Type'].str.contains(type_search, case=False, na=False)
    ]
monthly_df, dow_df, rev_month_df, heat_df = summarize_parking(df_filtered)

with visuals:
    # --- Top KPIs ---
    st.subheader('Key Performance Indicators')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Total Sessions', len(df_filtered))

    with col2:
        st.metric('Total Revenue ($)', df_filtered['Amount'].sum())

    with col3:
        dur = pd.to_numeric(df_filtered['duration'], errors='coerce')
        st.metric('Avg Duration (min)', round(dur.mean(), 2) if len(dur) > 0 else 0)

    st.markdown('---')

    v1, v2 = st.columns(2)
    v3, v4 = st.columns(2)

    with v1:
        # --- Monthly Sessions Trend ---
        monthly_fig = px.line(
            monthly_df,
            x='month',
            y='sessions',
            title='Monthly Parking Sessions (Total Count)',
            markers=True,
            category_orders={'month': month_order}
        )
        monthly_fig.update_layout(height=300)
        st.plotly_chart(monthly_fig, width='stretch')
        st.caption('Shows how overall parking activity changes month by month. Higher counts in summer reflect Banff’s peak tourist season, while lower numbers in winter show reduced visitor traffic.')

    with v2:
    # --- Day of Week Trend ---
        dow_fig = px.bar(
            dow_df,
            x='day_of_week',
            y='sessions',
            title='Parking Sessions by Day of Week (Overall)',
            category_orders={'day_of_week': weekday_order}
        )
        dow_fig.update_layout(height=300)
        st.plotly_chart(dow_fig, width='stretch')
        st.caption('Compares total parking sessions across the week. Weekend peaks suggest increased tourism and leisure visits, while weekdays remain steadier with local traffic.')

    with v3:
        # --- Monthly Revenue Trend ---
        rev_month_fig = px.line(
            rev_month_df,
            x='month',
            y='Amount',
            title='Total Revenue by Month ($)',
            markers=True,
            category_orders={'month': month_order}
        )
        rev_month_fig.update_layout(height=300)
        st.plotly_chart(rev_month_fig, width='stretch')
        st.caption('Tracks total monthly revenue from parking fees. Revenue rises in parallel with parking sessions, emphasizing the financial impact of seasonal visitor trends.')

    with v4:
        # --- Hour × Day-of-Week Heatmap ---
        pivot_table = heat_df.pivot(index='day_of_week', columns='hour', values='sessions').fillna(0)

        heatmap_fig = px.imshow(
            pivot_table,
            labels=dict(x='Hour of Day', y='Day of Week', color='Sessions'),
            title='Hourly Parking Activity by Day of Week',
            aspect='auto',
            color_continuous_scale='Blues'
        )
        heatmap_fig.update_layout(height=300)
        st.plotly_chart(heatmap_fig, width='stretch')
        st.caption('Shows the distribution of parking sessions across the day. Higher counts indicate more visitors during peak hours.')


# --- Footer ---
st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
           
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")
