"""
Hyperparameter tuning pipeline for the Banff Route-Level Traffic Management project.
Performs RandomizedSearchCV using TimeSeriesSplit per route for both RandomForest and XGBoost models.
"""

# --- Imports ---
import os
import joblib
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tqdm import tqdm


# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Load config ---
def load_config(config_path: str = 'configs/tune_config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# --- Main execution ---
def main():
    config = load_config()

    required_keys = ['data', 'search', 'mlflow', 'RandomForest', 'XGBoost']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f'Missing keys in tune_config.yaml: {missing}')

    DATA_PATH = config['data']['path']
    SAVE_DIR = config['data']['model_dir']
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- MLflow setup ---
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # --- Load data ---
    logger.info(f'Loading preprocessed data from {DATA_PATH}')
    df = joblib.load(DATA_PATH)

    route_col = config['data']['route_col']
    time_col = config['data']['time_col']
    target = config['data']['target']

    routes = (
        sorted(df[route_col].unique().tolist())
        if config['data']['routes'] == 'all'
        else config['data']['routes']
    )

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=config['data']['n_splits'])

    rf_params = config['RandomForest']
    xgb_params = config['XGBoost']

    logger.info(f'Tuning {len(routes)} route models using TimeSeriesSplit({config["data"]["n_splits"]})...')

    results = []

    # --- Parent MLflow run ---
    with mlflow.start_run(run_name='Banff_Tuning_Master', nested=False):
        for route in tqdm(routes, desc='Tuning models by route'):
            route_df = (
                df[df[route_col] == route]
                .sort_values(time_col)
                .reset_index(drop=True)
                .copy()
            )

            if route_df.empty:
                logger.warning(f'⚠️ Route {route} has no data, skipping.')
                continue

            X = route_df.drop(columns=config['data']['features_to_drop'])
            y = route_df[target]

            logger.info(f'🔹 Route {route}: {X.shape[0]} samples, {X.shape[1]} features')

            # --- Random Forest ---
            with mlflow.start_run(run_name=f'{route}_RandomForest', nested=True):
                logger.info(f'Starting RandomForest tuning for route {route}')
                rf = RandomForestRegressor(random_state=config['search']['random_state'])

                rf_search = RandomizedSearchCV(
                    estimator=rf,
                    param_distributions=rf_params,
                    n_iter=config['search']['n_iter'],
                    cv=tscv,
                    scoring=scorer,
                    verbose=1,
                    random_state=config['search']['random_state'],
                    n_jobs=config['search']['n_jobs']
                )
                rf_search.fit(X, y)

                best_rf = rf_search.best_estimator_
                best_params = rf_search.best_params_
                best_mae = -rf_search.best_score_  # negate because MAE is negated

                mlflow.log_params(best_params)
                mlflow.log_metric('best_cv_mae', best_mae)
                mlflow.sklearn.log_model(best_rf, artifact_path=f'{route}_rf_best')

                model_path = os.path.join(SAVE_DIR, f'{route}_rf_best.pkl')
                joblib.dump(best_rf, model_path)
                logger.info(f'Saved best RF model for {route} with MAE={best_mae:.3f}')

                results.append({
                    'route': route,
                    'model': 'RandomForest',
                    'mae': best_mae,
                    **best_params
                })

            # --- XGBoost ---
            with mlflow.start_run(run_name=f'{route}_XGBoost', nested=True):
                logger.info(f'Starting XGBoost tuning for route {route}')
                xgb = XGBRegressor(random_state=config['search']['random_state'])

                xgb_search = RandomizedSearchCV(
                    estimator=xgb,
                    param_distributions=xgb_params,
                    n_iter=config['search']['n_iter'],
                    cv=tscv,
                    scoring=scorer,
                    verbose=1,
                    random_state=config['search']['random_state'],
                    n_jobs=config['search']['n_jobs']
                )
                xgb_search.fit(X, y)

                best_xgb = xgb_search.best_estimator_
                best_params = xgb_search.best_params_
                best_mae = -xgb_search.best_score_

                mlflow.log_params(best_params)
                mlflow.log_metric('best_cv_mae', best_mae)
                mlflow.sklearn.log_model(best_xgb, artifact_path=f'{route}_xgb_best')

                model_path = os.path.join(SAVE_DIR, f'{route}_xgb_best.pkl')
                joblib.dump(best_xgb, model_path)
                logger.info(f'Saved best XGB model for {route} with MAE={best_mae:.3f}')

                results.append({
                    'route': route,
                    'model': 'XGBoost',
                    'mae': best_mae,
                    **best_params
                })

    # --- Save tuning summary ---
    summary_path = os.path.join(SAVE_DIR, 'tuning_summary.csv')
    pd.DataFrame(results).to_csv(summary_path, index=False)
    logger.info(f'Saved tuning summary to {summary_path}')
    logger.info('Hyperparameter tuning complete for all routes!')


if __name__ == '__main__':
    main()
