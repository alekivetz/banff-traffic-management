import re
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from utils.display_images import display_banner
from utils.data_loader import fetch_routes_data, fetch_parking_vis_chatbot


st.markdown("""
<style>
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 100;
    }
    .main .block-container { padding-bottom: 8rem; }
    .stChatMessage { max-height: 70vh; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)


# --- Context Documents ---
project_overview = """
The Banff Traffic Management project supports smarter mobility within the town of Banff.
It predicts overall and route-level traffic congestion and parking occupancy, using Random Forest and XGBoost models trained on historical traffic and parking data.
The system is designed to help park planners anticipate congestion, manage parking demand, and support sustainable tourism.
"""

model_explainability = """
SHAP values are used to explain how the traffic forecasting models predict congestion across Banff’s routes.
They measure how much each feature — such as travel speed, time of day, or recent delay trends — contributes to the model’s final prediction.
By visualizing these effects, stakeholders can see whether high delays are driven by peak hours, slow speeds, or route-specific factors.
This transparency builds trust in the model and supports data-driven decision making for congestion management.
"""

model_explainability = """
Model explainability uses SHAP values to interpret how machine learning models make predictions.
They quantify how much each feature contributes to the model’s final prediction.
By analyzing feature importance, it identifies the most influential factors driving congestion and parking demand.
This helps stakeholders understand *why* a prediction was made, improving transparency and trust in the system.
"""

traffic_forecaster = """
The Traffic Forecasting component predicts overall congestion risk and route-specific delay durations.
Forecasters are generated using Random Forest regressors and classifiers trained on historical data and lagged features. 
"""

parking_forecaster = """
The Parking Forecasting component tracks occupancy, duration, and revenue trends across Banff parking facilities.
Forecasts are generated in 15-minute intervals using historical transaction data and seasonal patterns.
Insights from this analysis help identify high-demand periods, optimize parking availability, and inform sustainable transportation planning.
"""

model_performance = """
**Traffic Forecasting**\n
Per-route Random Forest regressors achieved:
• **R²:** 0.862
• **MAE:** 0.023 minutes  
• **RMSE:** 0.536 minutes  

---

**Congestion Classification**  
A Random Forest classifier categorized congestion into *Low*, *Medium*, and *High* classes.  

**Performance metrics:**  
• **Accuracy:** 0.70  
• **Macro Recall:** 0.52  
• **Weighted F1-Score:** 0.79  

The model performs best for low-congestion routes and captures high congestion with **65% recall**.

---

**Parking Forecasting**  
An XGBoost regression model predicted parking occupancy and duration.  

**Results:**  
• **R²:** 0.95  
• **MAE:** 0.97   
"""


data_sources = """
This project uses historical traffic and parking data from the town of Banff's traffic sensors and parking systems.
Route datasets include information on timestamps, travel speeds, and delays.
Parking datasets include unit IDs, payment information, and parking start/end timestamps. 
"""

documents = {
    'overview': project_overview,
    'explainability': model_explainability,
    'parking': parking_forecaster,
    'traffic': traffic_forecaster,
    'performance': model_performance,
    'data': data_sources
}

# --- Embeddings + Generator (RAG) ---
@st.cache_resource
def load_embeddings():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = {k: embedder.encode(v, convert_to_tensor=True) for k, v in documents.items()}
    return embedder, doc_embeddings

@st.cache_resource
def load_generator():
    """Load lightweight FLAN-T5 generator safely across environments."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return pipeline(
        'text2text-generation',
        model='google/flan-t5-small',
        device=0 if device == 'cuda' else -1
    )

def retrieve_context(query, top_k=2):
    q_lower = query.lower()
    
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {k: util.pytorch_cos_sim(query_emb, emb).item() for k, emb in doc_embeddings.items()}
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return '\n\n---\n\n'.join(f'{d.upper()} SECTION:\n{documents[d]}' for d, _ in top_docs)

# --- Intent Matching ---
@st.cache_resource
def load_intent_imbedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

intent_embedder = load_intent_imbedder()

# Predefined intent categories

INTENTS = {
    'parking': [
        'parking duration',
        'how long people park',
        'parking sessions',
        'which parking lot',
        'banff parking',
        'average stay',
        'parking occupancy',
        'busiest parking lot'
    ],
    'traffic': [
        'traffic delay',
        'vehicle speed',
        'route congestion',
        'road conditions',
        'travel time',
        'rush hour',
        'busiest routes',
        'traffic volume'
    ],
    'project': [
        'project overview',
        'machine learning model',
        'random forest',
        'xgboost',
        'explainability',
        'model training',
        'shap values',
        'feature importance'
    ],
    'models': [
        'model performance', 
        'how did the models perform', 
        'metrics', 
        'performance metrics',
        'results summary', 
        'forecasting results', 
        'performance summary', 
        'model results'
    ]
}

# Encode all examples once
INTENT_EMBEDDINGS = {
    k: intent_embedder.encode(v, convert_to_tensor=True) for k, v in INTENTS.items()
}

def detect_intent(query):
    """Return best-matching intent based on cosine similarity."""
    q_emb = intent_embedder.encode(query, convert_to_tensor=True)
    scores = {
        intent: float(util.pytorch_cos_sim(q_emb, emb).max())
        for intent, emb in INTENT_EMBEDDINGS.items()
    }
    best_intent = max(scores, key=scores.get)
    return best_intent if scores[best_intent] > 0.35 else 'project'  # fallback to RAG

def query_llm(query, context):
    prompt = (
        'You are a helpful assistant for the Banff Traffic Management project.\n\n'
        'Always return answers using **Markdown formatting** - with clear section headers, bullet points, and line breaks. '
        'Do NOT merge multiple sections into one paragraph; preserve all newlines.\n\n'
        f'Context:\n{context}\n\n'
        f'User Query: {query}\n\n'
        'Answer (in Markdown):'
    )

    output = generator(
        prompt,
        max_new_tokens=400,
        do_sample=False,
        temperature=0.7,
        truncation=True
    )

    # Extract the generated text and preserve formatting
    reply = output[0]['generated_text'].strip()
    reply = reply.replace('\\n', '\n').replace('  ', ' ')
    return reply


# --- Rule-Based Data Queries ---
def extract_iso_date(query):
    match = re.search(r'\d{4}-\d{2}-\d{2}', query)
    return match.group(0) if match else None

# --- Parking Queries
def handle_parking_query(q):
    df = parking_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Starting date']):
        df['Starting date'] = pd.to_datetime(df['Starting date'], errors='coerce')

    q_lower = q.lower()
    date_str = extract_iso_date(q_lower)

    # Count sessions by day
    if date_str and any(k in q_lower for k in ['how many', 'how often', 'number of']):
        count = len(df[df['Starting date'].dt.date == pd.to_datetime(date_str).date()])
        return f'On {date_str}, there were {count} parking sessions.'

    # Average duration by day   
    if date_str and 'average' in q_lower and 'duration' in q_lower:
        filtered = df[df['Starting date'].dt.date == pd.to_datetime(date_str).date()]
        if filtered.empty:
            return f'No parking sessions found on {date_str}.'
        avg = filtered['duration'].mean()
        return f'On {date_str}, the average parking duration was {avg:.2f} hours.'

    # Average duration for specific unit
    match = re.search(r'(banff\d+)', q_lower)
    if match and 'average' in q_lower and 'duration' in q_lower:
        key = match.group(1)
        mask = df['Unit'].astype(str).str.contains(key, case=False, na=False)
        filtered = df[mask]
        if filtered.empty:
            return f'No sessions found for unit matching "{key}".'
        avg = filtered['duration'].mean() / 60
        unit_name = filtered['Unit'].iloc[0]
        return f'{unit_name} has {len(filtered)} sessions with an average duration of {avg:.2f} hours.'

    # Longest average duration (unit/lot synonyms)
    if any(k in q_lower for k in ['unit', 'lot', 'location', 'parkade', 'parking lot']) and \
       any(k in q_lower for k in ['longest', 'longer', 'highest', 'most time']):
        df['duration_hours'] = df['duration'] / 60
        grouped = df.groupby('Unit')['duration_hours'].agg(['mean', 'count']).reset_index()
        row = grouped.loc[grouped['mean'].idxmax()]
        
        return (
            f'The parking lot with the longest average duration is **{row["Unit"]}**, '
            f'with an average stay of {row["mean"]:.2f} hours.'
        )
    
    # Overall average parking duration - no date or unit
    if 'average' in q_lower and 'duration' in q_lower:
        avg = df['duration'].mean() / 60
        total = len(df)
        return (
            f'The overall average parking duration across all sessions '
            f'is {avg:.2f} hours.'
        )
    return None  # No match found

# --- Traffic Queries
def handle_traffic_query(q):
    df = routes_df.copy()
    q_lower = q.lower()

    # Delay analysis 
    if any(k in q_lower for k in ['delay', 'travel time']):
        max_delay = df['actual_delay'].max()
        min_delay = df['actual_delay'].min()
        avg_delay = df['actual_delay'].mean()
        return (
            f'**Traffic Delay Summary**  \n'
            f'- **Average Delay:** {avg_delay:.2f} minutes  \n'
            f'- **Maximum Delay:** {max_delay:.2f} minutes  \n'
            f'- **Minimum Delay:** {min_delay:.2f} minutes'
        )


    # Speed analysis
    if any(k in q_lower for k in ['speed', 'velocity', 'pace']):
        max_speed = df['speed_kmh'].max()
        min_speed = df['speed_kmh'].min()
        avg_speed = df['speed_kmh'].mean()
        return (
            f'**Speed Summary**  \n'
            f'- **Average Speed:** {avg_speed:.1f} km/h  \n'
            f'- **Highest Speed:** {max_speed:.1f} km/h  \n'
            f'- **Lowest Speed:** {min_speed:.1f} km/h'
        )

    # Most congested routes
    if any(k in q_lower for k in ['busiest', 'most congested', 'worst traffic', 'highest congestion']):
        busiest = df.groupby('route')['actual_delay'].mean().nlargest(3)
        formatted = '\n'.join([f'- **{route}**: {delay:.2f} min avg delay' for route, delay in busiest.items()])
        return (
            f'**Most Congested Routes**  \n'
            f'{formatted}'
        )


    return None


# --- Routing Logic ---
def get_bot_response(query):
    intent = detect_intent(query)

    if intent == 'parking':
        resp = handle_parking_query(query)
        if resp:
            return resp

    if intent == 'traffic':
        resp = handle_traffic_query(query)
        if resp:
            return resp

    if intent == 'models':
        return model_performance
    
    # fallback: project or general
    context = retrieve_context(query)
    return query_llm(query, context)


# --- Page Config & Style ---
st.set_page_config(page_title='Banff Project Chatbot', page_icon='💬', layout='wide')
display_banner()

# --- Data and resource loading ---
with st.spinner('Loading chatbot data...'):
    try:
        routes_df = fetch_routes_data()
        parking_df = fetch_parking_vis_chatbot()
    except Exception as e:
        st.error(f'Could not load chatbot data: {e}')
        st.stop()

with st.spinner('Loading chatbot resources...'):
    try: 
        embedder, doc_embeddings = load_embeddings()
        generator = load_generator()
    except Exception as e:
        st.error(f'Could not load chatbot resources: {e}')


st.title('Banff Traffic Management Chatbot')
st.write("""
Ask questions about **traffic conditions**, **parking analytics**, or the **project itself**.  
This chatbot blends **retrieval-augmented generation (RAG)** with **data-driven insights** to provide both factual and conceptual answers.
""")

# --- Example Queries ---
# Show expanded only before first chat
expanded_state = not st.session_state.get('messages')

with st.expander("💡 Try asking questions like these:", expanded=expanded_state):
    st.markdown("""
    Ask about **parking**, **traffic**, or **project details** to explore insights powered by machine learning.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🅿️ Parking Queries
        - What is the average parking duration?  
        - How many parking sessions were there on **2025-08-31**?  
        - What is the average parking duration for **BANFF01**?  
        - Which unit has the **longest average duration**?
        """)

    with col2:
        st.markdown("""
        ### 🚗 Traffic Queries
        - What is the **average delay** across all routes?  
        - What’s the **maximum speed** recorded on Route 3?  
        - Show me the **delay trends** for November.  
        - Which routes have the **highest congestion**?
        """)

    with col3:
        st.markdown("""
        ### 📊 Project & Model Queries
        - How does the Banff Traffic Management project work?  
        - What algorithms are used for prediction?  
        - How do **SHAP values** explain congestion predictions?  
        - How did the models perform?  
        """)


st.markdown('---')


# --- Chat UI ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if prompt := st.chat_input('Ask me about Banff traffic, parking, or our Banff Traffic Management project...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner('Analyzing your question...'):
            reply = get_bot_response(prompt)
            st.markdown(reply, unsafe_allow_html=True)

    st.session_state.messages.append({'role': 'assistant', 'content': reply})