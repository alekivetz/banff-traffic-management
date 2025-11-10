"""
Preprocessing pipeline for the Banff Traffic Management project. 
Handles data loading, cleaning, and global feature engineering.
"""

# --- Standard library imports ---
import argparse
import logging
import os
from typing import Tuple

# --- Third-party imports ---
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# --- Load config ---
def load_config(config_path: str = 'configs/preprocess_config.yaml') -> dict:
    '''Load preprocessing configuration from YAML file.'''
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# --- Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Data loading ---
def load_data(filepath: str) -> pd.DataFrame:
    '''Load data from CSV file.'''
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f'Loaded data from {filepath} - shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Failed to load data: {e}')
        raise


# --- Rename and drop columns ---
def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    '''Rename columns based on YAML mapping.'''
    missing = [col for col in rename_map.keys() if col not in df.columns]
    if missing:
        logger.warning(f'Columns not found for renaming: {missing}')
    df = df.rename(columns=rename_map)
    logger.info(f'Renamed columns: {list(rename_map.keys())}')
    return df


def drop_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    '''Drop columns based on YAML list.'''
    existing = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing)
    logger.info(f'Dropped columns: {existing}')
    return df


# --- Data cleaning ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''Performs basic data cleaning.'''
    df = df.copy()

    # Convert timestamp to datetime
    df['timestamp'] = df['timestamp'].str.rsplit(' ', n=1).str[0]
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %Y %H:%M:%S', errors='coerce')
    df = df.sort_values(['route', 'timestamp']).reset_index(drop=True)

    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()
    logger.info(f'Dropped {duplicate_count} duplicate rows')

    # Convert numeric columns
    for col in ['speed', 'delay', 'mean_travel_time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing critical values
    missing_subset = df[df[['speed', 'delay']].isna().any(axis=1)]
    df = df.drop(missing_subset.index)
    logger.info(f'Dropped {missing_subset.shape[0]} rows with missing values')

    return df


# --- Remove outliers ---
def remove_delay_outliers(df: pd.DataFrame) -> pd.DataFrame:
    '''Remove extreme delay outliers above the 99.9th percentile.'''
    df = df.copy()
    cleaned_routes = []

    for route in df['route'].unique():
        route_df = df[df['route'] == route]
        threshold = route_df['delay'].quantile(0.999)
        filtered = route_df[route_df['delay'] <= threshold]
        removed_count = len(route_df) - len(filtered)
        logger.info(f'Removed {removed_count} outliers for route {route}')
        cleaned_routes.append(filtered)

    df_cleaned = pd.concat(cleaned_routes, ignore_index=True)
    return df_cleaned


# --- Feature engineering ---
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Performs feature engineering on the route data.
        - time-based features (hour, day_of_week, is_weekend)
    '''
    df = df.copy()

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    logger.info('Feature engineering complete')
    return df


# --- Encoding ---
def one_hot_encode(df: pd.DataFrame, encode_map: dict) -> pd.DataFrame:
    '''Encode categorical features using one-hot encoding.'''
    df = df.copy()

    if not encode_map:
        logger.info('No categorical features to encode')
        return df

    one_hot_cols = encode_map.get('one_hot', [])
    if one_hot_cols:
        if isinstance(one_hot_cols, str):
            one_hot_cols = [one_hot_cols]
        for col in one_hot_cols:
            if col not in df.columns:
                logger.warning(f'Column {col} not found for one-hot encoding')
                continue
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            df.columns = df.columns.str.lower().str.strip()
            logger.info(f'Encoded {col} using one-hot encoding')

    return df


# --- Preprocessing pipeline ---
def preprocess_pipeline(filepath: str, config: dict) -> pd.DataFrame:
    '''Full preprocessing pipeline for Banff Traffic Management data.'''
    logger.info(f'Starting preprocessing pipeline for {filepath}')

    RENAME_FEATURES = config['features']['rename']
    ENCODE_FEATURES = config['features']['encode']
    DROP_COLUMNS = config['features']['drop']

    df = load_data(filepath)
    df = rename_columns(df, RENAME_FEATURES)
    df = drop_columns(df, DROP_COLUMNS)
    df = clean_data(df)
    df = remove_delay_outliers(df)
    df = feature_engineering(df)
    df = one_hot_encode(df, ENCODE_FEATURES)

    logger.info(f'Preprocessing pipeline complete - final dataset shape: {df.shape}')
    return df


# --- Save data ---
def save_preprocessed_data(df: pd.DataFrame, output_dir: str) -> dict:
    '''Save preprocessed data to disk.'''
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    joblib_path = os.path.join(output_dir, 'banff_route_preprocessed.pkl')
    joblib.dump(df, joblib_path)
    saved_paths['joblib'] = joblib_path
    logger.info(f'Saved preprocessed data with joblib to {joblib_path}')

    csv_path = os.path.join(output_dir, 'banff_route_data_preprocessed.csv')
    df.to_csv(csv_path, index=False)
    saved_paths['csv'] = csv_path
    logger.info(f'Saved CSV to {csv_path}')

    return saved_paths


# --- Main ---
def main():
    '''Main CLI entry point for preprocessing.'''
    parser = argparse.ArgumentParser(description='Preprocess route data for Banff Traffic Management model training.')

    parser.add_argument('--config', type=str, default='configs/preprocess_config.yaml', help='Path to configuration file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the raw CSV file.')
    parser.add_argument('--output-dir', type=str, help='Directory to save preprocessed data.')

    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config['data']['processed_path']

    logger.info(f'Preprocessing data from {args.input} using config: {args.config}')

    df = preprocess_pipeline(args.input, config)
    saved_paths = save_preprocessed_data(df, output_dir)

    print('\n' + '=' * 60)
    print('Preprocessing completed successfully!')
    print('=' * 60)
    print(f'Final dataset shape: {df.shape}')
    print('\nSaved files:')
    for key, path in saved_paths.items():
        print(f'  - {key}: {path}')
    print('=' * 60)

    logger.info('Preprocessing completed!')


if __name__ == '__main__':
    main()
