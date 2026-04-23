import io
import joblib
import json
import streamlit as st
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# --- Connect to Google Drive ---
@st.cache_resource
def connect_drive():
    """Authenticate with Google Drive using the service account."""
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets['gcp_service_account']),
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)

def download_from_drive(file_id):
    """Download a file from Google Drive."""
    drive = connect_drive()
    request = drive.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    buffer.seek(0)
    return buffer.read()

# --- Generic Parquet Data Loader ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_parquet(file_id):

    try:
        data_bytes = download_from_drive(file_id)
        df = pd.read_parquet(io.BytesIO(data_bytes), engine='fastparquet')
    except Exception as e:
        st.warning(f'Could not load from Google Drive ({e}).')
    return df

# --- Load Datasets ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_routes_data(for_models=False):
    """Load routes dataset for ML models, visualization dashboards, and chatbot."""
    df = fetch_parquet(st.secrets['ROUTES_DATA_ID'])
    if for_models:
        return df
    else:
        df = df.drop(columns=['month', 'day_of_week'], errors='ignore')
        df['month'] = df['timestamp'].dt.strftime('%B')
        df['day_of_week'] = df['timestamp'].dt.day_name()
        cols_to_keep = ['route', 'timestamp', 'speed_kmh', 
                        'mean_travel_time', 'actual_delay',
                        'day_of_week', 'hour', 'month']
        return df[cols_to_keep]

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_parking_vis_chatbot():
    """Parking dataset for visual dashboards and chatbot."""
    return fetch_parquet(st.secrets['PARKING_VIS_CHATBOT_ID'])


# # --- Model Loaders ---
@st.cache_resource(show_spinner=False)
def fetch_classifier():
    file_id = st.secrets['CLASSIFIER_MODEL_ID']
    model_bytes = download_from_drive(file_id)
    return joblib.load(io.BytesIO(model_bytes))


@st.cache_resource(show_spinner=False)
def fetch_single_regressor(route: str):
    """Load only one route-specific regressor model."""
    route_dict = json.loads(st.secrets['ROUTE_MODELS_IDS'])
    if route not in route_dict:
        raise ValueError(f"No model ID found for route: {route}")
    file_id = route_dict[route]
    model_bytes = download_from_drive(file_id)
    return joblib.load(io.BytesIO(model_bytes))

@st.cache_resource(show_spinner=False)
def fetch_regressors():
    """
    Load ALL per-route regressors from Google Drive into memory.
    """
    route_dict = json.loads(st.secrets['ROUTE_MODELS_IDS'])
    regressors = {}

    for route, file_id in route_dict.items():
        try:
            model_bytes = download_from_drive(file_id)
            regressors[route] = joblib.load(io.BytesIO(model_bytes))
        except Exception as e:
            st.warning(f"Could not load model for {route}: {e}")

    return regressors

@st.cache_data(ttl=86400, show_spinner=False)
def load_parking_resources():
    """Load the parking model, encoder, and modeling dataset from Google Drive."""
    try:
        model = joblib.load(io.BytesIO(download_from_drive(st.secrets['PARKING_MODEL_ID'])))
        encoder = joblib.load(io.BytesIO(download_from_drive(st.secrets['PARKING_ENCODER_ID'])))
        df = fetch_parquet(st.secrets['PARKING_DATA_ID'])
        return model, encoder, df
    except Exception as e:
        st.warning(f'Failed to load parking resources: {e}')
        return None, None, pd.DataFrame()

