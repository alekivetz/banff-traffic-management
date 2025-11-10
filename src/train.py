"""
Training pipeline for the Banff Traffic Management project.
Trains per-route regression models (Random Forest and XGBoost)
using TimeSeriesSplit cross-validation and chronological holdouts.

Now automatically loads best hyperparameters from tuning_sumamary.csv if it exists.
"""

# Standard library imports
import logging
import os
from math import sqrt

# Third-party imports
import joblib   
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tqdm import tqdm   


# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Load config ---

def load_config(
        config_path: str = 'configs/train_config.yaml'
    ) -> dict:
    """Load train configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# --- Load preprocessed data ---
    
def load_preprocessed_data(
        filepath: str
    ) -> pd.DataFrame:
    """Load preprocessed data (CSV or joblib)."""
    
    logger.info(f'Loading preprocessed data from {filepath}')
    if filepath.endswith('csv'):
        df = pd.read_csv(filepath)
    else:
        df = joblib.load(filepath)

    logger.info(f'Loaded preprocessed data - shape: {df.shape}')
    return df

# --- Load best hyperparameters from tuning summary ---
def load_tuned_params(summary_path: str) -> dict:
    if not os.path.exists(summary_path):
        logger.warning(f'No tuning summary found at {summary_path}, using defaults from config.')
        return {}

    df = pd.read_csv(summary_path)
    tuned = {}

    for _, row in df.iterrows():
        route = row['route']
        model = row['model']
        # Filter out non-param columns
        params = {k: row[k] for k in df.columns if k not in ['route', 'model', 'mae']}
        tuned.setdefault(route, {})[model] = params

    logger.info(f'Loaded tuned hyperparameters for {len(tuned)} routes from {summary_path}')
    return tuned


# --- Prepare per-route CV/Holdout splits and split-dependent features ---

def prepare_route_datasets(
        df: pd.DataFrame,
        config: dict
    ):
    """
    Prepares per-route datasets with engineered lag and rolling features for
    time-series model training and evaluation. 
    """

    TARGET = config['target']
    TIME_COL = config['time_col']
    num_splits = config['n_splits']
    lag_list = config['lag_list']
    roll_windows = config['roll_windows']

    # Containers
    X_train_cv, y_train_cv, X_test_holdout, y_test_holdout, tscv_splits = {}, {}, {}, {}, {}

    sorted_routes = sorted(df['route'].unique().tolist())

    def is_engineered(name: str) -> bool:
        prefixes = ('delay_lag_', 'mean_travel_time_lag_', 'delay_roll_', 'mean_travel_time_roll_')
        return name.startswith(prefixes)
    
    # Base features - everything except target/time/route and engineered
    base_features = [
        col for col in df.columns if col not in {TARGET, TIME_COL, 'route'} and not is_engineered(col)
    ]

    for route in sorted_routes: 
        route_data = df[df['route'] == route].sort_values(TIME_COL).copy()
        n = len(route_data)

        # Skip small routese
        if n < max(roll_windows) + 10:
            continue

        # Chronological 80/20 split
        split_idx = int(n * 0.8)    
        cv_data = route_data.iloc[:split_idx].copy()
        hold_data = route_data.iloc[split_idx:].copy()

        # Create lags
        for i in lag_list:
            for col in [TARGET, 'mean_travel_time']:
                cv_data[f'{col}_lag_{i}'] = cv_data[col].shift(i)
                hold_data[f'{col}_lag_{i}'] = hold_data[col].shift(i)

        # Create rolling windows
        for w in roll_windows: 
            for col in [TARGET, 'mean_travel_time']:
                for agg in ['mean', 'max']:
                    cv_data[f'{col}_roll_{agg}_{w}'] = cv_data[col].rolling(w, min_periods=w).agg(agg)
                    hold_data[f'{col}_roll_{agg}_{w}'] = hold_data[col].rolling(w, min_periods=w).agg(agg)

        # Drop any NANs created
        engineered_cols = [col for col in cv_data.columns if is_engineered(col)]
        cv_data = cv_data.dropna(subset=engineered_cols)
        hold_data = hold_data.dropna(subset=engineered_cols)

        # Ensure enough samples remain
        if len(cv_data) < 20 or len(hold_data) < 5:
            continue

        # Final features
        feature_cols = base_features + engineered_cols

        # Store holdout
        X_test_holdout[route] = hold_data[feature_cols].copy()
        y_test_holdout[route] = hold_data[TARGET].copy()

        # Store CV data with folds
        X_cv, y_cv = cv_data[feature_cols].copy(), cv_data[TARGET].copy()
        tscv = TimeSeriesSplit(n_splits=num_splits)
        X_train_cv[route], y_train_cv[route], tscv_splits[route] = X_cv, y_cv, list(tscv.split(X_cv))

    logger.info(f'Prepared {len(X_train_cv)} routes for training')

    return X_train_cv, y_train_cv, X_test_holdout, y_test_holdout, tscv_splits

# --- Train models ---

def train_per_route_models(
        X_train_cv: dict,
        y_train_cv: dict,
        X_test_holdout: dict,
        y_test_holdout: dict,
        tscv_splits: dict,
        config: dict,
        tuned_params: dict = None,
        save_dir: str='models'
    ):
    """
    Train RandomForest and XGBoost models independently per route using time-series cross validation
    and a chronological holdout (test) set. 
    """

    os.makedirs(save_dir, exist_ok=True)
    model_results = []

    for route in tqdm(X_train_cv.keys(), desc='Training models by route'):
        X_cv, y_cv = X_train_cv[route], y_train_cv[route]
        X_test, y_test = X_test_holdout[route], y_test_holdout[route]
        folds = tscv_splits[route]

        # Load tuned params if available
        rf_params = tuned_params.get(route, {}).get('RandomForest', config['models']['RandomForest'])
        xgb_params = tuned_params.get(route, {}).get('XGBoost', config['models']['XGBoost'])


        models = {
            'RandomForest': RandomForestRegressor(**config['models']['RandomForest'], n_jobs=-1),
            'XGBoost': XGBRegressor(**config['models']['XGBoost'], n_jobs=-1)
        }

        for model_name, model in models.items():
            fold_scores = []

            # Cross-validation
            for train_idx, val_idx in folds:
                X_train, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
                y_train, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                fold_scores.append((mae, rmse))

            avg_mae, avg_rmse = np.mean(fold_scores, axis=0)
            holdout_preds = model.predict(X_test)
            holdout_mae = mean_absolute_error(y_test, holdout_preds)
            holdout_rmse = np.sqrt(mean_squared_error(y_test, holdout_preds))

            result = {
                'Route': route, 
                'Model': model_name,
                'CV MAE': avg_mae,
                'CV RMSE': avg_rmse,
                'Holdout MAE': holdout_mae,
                'Holdout RMSE': holdout_rmse
            }
            model_results.append(result)

            # Log to MLflow
            with mlflow.start_run(run_name=f'{route}_{model_name}', nested=True):   
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(
                    {
                        'cv_mae': avg_mae,
                        'cv_rmse': avg_rmse,
                        'holdout_mae': holdout_mae,
                        'holdout_rmse': holdout_rmse
                    }
                )
                mlflow.sklearn.log_model(model, artifact_path=f'{route}_{model_name}')

            # Save model locally
            model_path = os.path.join(save_dir, f'{route}_{model_name}.pkl')
            joblib.dump(model, model_path)

    comparison_df = pd.DataFrame(model_results).sort_values(
        by=['Route', 'Holdout MAE']
    )

    comparison_df.to_csv(os.path.join(save_dir, 'route_model_comparison.csv'), index=False)
    joblib.dump(comparison_df, os.path.join(save_dir, 'route_model_comparison.pkl'))

    logger.info(f'Trained and logged {len(comparison_df)} models to MLflow')
    return comparison_df

# --- Save combined holdout set ---

def save_combined_holdout(X_test_holdout: dict, y_test_holdout: dict, output_path: str):
    """
    Combine all per-route holdout sets into one DataFrame and save as Parquet.
    This file will be used later by evaluate.py.
    """
    logger.info('Combining route holdout datasets for evaluation...')
    holdout_frames = []

    for route, X_hold in X_test_holdout.items():
        y_hold = y_test_holdout[route]
        temp = X_hold.copy()
        temp['actual_delay'] = y_hold.values
        temp['route'] = route
        holdout_frames.append(temp)

    combined_df = pd.concat(holdout_frames, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_parquet(output_path, index=False)
    logger.info(f'Saved combined holdout dataset to {output_path} (shape={combined_df.shape})')

    return combined_df


# --- Main execution ---

def main():
    config = load_config()

    # Initialize MLflow experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


    df = load_preprocessed_data(config['data']['path'])
    tuned_params = load_tuned_params(os.path.join(config['data']['model_dir'], 'tuning_summary.csv'))

    X_train_cv, y_train_cv, X_test_holdout, y_test_holdout, tscv_splits = prepare_route_datasets(df, config['data'])
    
    comparison_df = train_per_route_models(
        X_train_cv, y_train_cv, X_test_holdout, y_test_holdout, tscv_splits, 
        config, tuned_params, save_dir=config['data']['model_dir'])
    
    # Save combined holdout for evaluation
    holdout_path = os.path.join(config['data']['processed_path'], 'route_holdout.parquet')
    save_combined_holdout(X_test_holdout, y_test_holdout, holdout_path)

    logger.info('Training complete. Results saved under models/ and logged to MLflow.')

if __name__ == '__main__':
    main()
