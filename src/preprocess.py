"""
Preprocessing pipeline for the Banff Traffic Management project. 
Handles data loading, cleaning, and global feature engineering.
"""

# Standard library imports
import argparse 
import logging
import os
from typing import Tuple

# Third-party imports
import joblib   
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# --- Load config --- 

def load_config(
        config_path: str = 'configs/preprocess_config.yaml'
    ) -> dict:
    """Load prepocessing configuration from YAML file."""

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# --- Logger --- 
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)    

# --- Data loading ---

def load_data(
        filepath: str,
    ) -> pd.DataFrame:
    """Load data from CSV file."""

    try: 
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f'Loaded data from {filepath} - shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Failed to load data: {e}')
        raise

# --- Rename and drop columns ---

def rename_columns(
        df: pd.DataFrame,
        rename_map: dict
    ) -> pd.DataFrame:
    """Rename columns based on YAML mapping."""

    missing = [col for col in rename_map.keys() if col not in df.columns]
    if missing: 
        logger.warning(f'Columns not found for renaming: {missing}')
    df = df.rename(columns=rename_map)
    logger.info(f'Renamed columns: {list(rename_map.keys())}')
    return df

def drop_columns(
        df: pd.DataFrame,
        drop_cols: list
    ) -> pd.DataFrame:
    """Drop columns based on YAML list."""

    existing = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing)
    logger.info(f'Dropped columns: {existing}')
    return df

# --- Data cleaning --- 

def clean_data(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """Performs basic data cleaning.""" 

    df = df.copy()  

    # Correct timestamp format, convert to datetime, and sort df by route and timestamp
    df['timestamp'] = (df['timestamp'].str.rsplit(' ', n=1).str[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %Y %H:%M:%S',errors='coerce')
    df = df.sort_values(['route', 'timestamp']).reset_index(drop=True)

    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()  
    logger.info(f'Dropped {duplicate_count} duplicate rows')

    # Missing values
    missing_subset = df[df[['speed', 'delay']].isna().any(axis=1)]
    df = df.drop(missing_subset.index)
    logger.info(f'Dropped {missing_subset.shape[0]} rows with missing values')  

    return df

# --- Remove outliers ---

def remove_delay_outliers(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
    """Remove extreme delay outliers, only above the 99.9th percentile."""
    
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

def feature_engineering(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Performs feature engineering on the route data. 
        - time-based features (hour, day_of_week, is_weekend)
        - derive distance feature from speed and travel time
    """

    df = df.copy()

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Distance
    # df['distance'] = df['speed'] * df['mean_travel_time']

    logger.info('Feature engineering complete')

    return df

# --- Encoding ---

def one_hot_encode(
        df: pd.DataFrame,
        encode_map: dict
    ) -> pd.DataFrame:
    """Encode categorical features using one-hot encoding."""

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

def preprocess_pipeline(
        filepath: str,
        config: dict
    ) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for the Banff Traffic Management route project.
    Executes data loading, cleaning, feature engineering, encoding and splitting.    
    """ 

    logger.info(f'Starting preprocessing pipeline for {filepath}')

    # Extract config parameters 
    RENAME_FEATURES      = config["features"]["rename"]
    ENCODE_FEATURES      = config["features"]["encode"]
    DROP_COLUMNS         = config["features"]["drop"]

    # Load data
    df = load_data(filepath)

    df = rename_columns(df, RENAME_FEATURES)
    df = drop_columns(df, DROP_COLUMNS)
    df = clean_data(df)

    # Outliers
    df = remove_delay_outliers(df)

    # Feature engineering
    df = feature_engineering(df)

    # Encoding
    df = one_hot_encode(df, ENCODE_FEATURES)

    logger.info(f'Preprocessing pipeline complete - final dataset shape: {df.shape}')

    return df

# --- Save data ---

def save_preprocessed_data(
        df: pd.DataFrame,
        output_dir: str
    ) -> dict:
    """Save preprocessed data to disk."""

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    # Save data with joblib
    joblib_path = os.path.join(output_dir, 'banff_route_preprocessed.pkl')
    joblib.dump(df, joblib_path)
    saved_paths['joblib'] = joblib_path
    logger.info(f"Saved preprocessed data with joblib to {joblib_path}")

    # Save CSV version 
    csv_path = os.path.join(output_dir, 'banff_route_data_preprocessed.csv')
    df.to_csv(csv_path, index=False)
    saved_paths['csv'] = csv_path
    logger.info(f'Saved CSV to {csv_path}')

    return saved_paths

# --- Main ---

def main():
    """
    Main function for Command Line Interface (CLI) preprocessing. 
    """

    # Load configuration
    config = load_config()

    parser = argparse.ArgumentParser(description='Preprocess route data for Banff Traffic Management model training.')

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the raw CSV file.'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=config["data"]["processed_path"],
        help=f"Directory to save preprocessed data (default: {config['data']['processed_path']})"
    )

    args = parser.parse_args()

    # Run preprocessing pipeline
    logger.info(f'Preprocessing data from {args.input}...')

    df = preprocess_pipeline(args.input, config)

    # Save preprocessed data
    saved_paths = save_preprocessed_data(df, args.output_dir)

    print("\n" + "="*60)
    print("Preprocessing completed successfully!")
    print("="*60)
    print(f'Final dataset shape: {df.shape}')
    print(f"\nSaved files:")
    for key, path in saved_paths.items():
        print(f"  - {key}: {path}")
    print("="*60)
    
    logger.info("Preprocessing completed!")


if __name__ == '__main__':
    main()

    
