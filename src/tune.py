"""
Hyperparemeter tuning pipeline for the Banff Route-Level Traffic Management project.
Performs RandomizedSearchCV using TimeSeriesSplit per route.
"""

# Standard library imports
import os

# Third-party imports
import joblib   
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


# --- Load config ---

def load_config(config_path: str ='configs/tune_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Main execution ---

def main():
    config = load_config()

    DATA_PAth = config['data']['path']
    SAVE_DIR = config['data']['model_dir']
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Setup MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load data
    df = joblib.load(DATA_PAth)
    routes = (
        sorted(df['route'].unique().tolist())
        if config['data']['routes'] == 'all'
        else config['data']['routes']
    )

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=config['data']['n_splits'])

    # Parameter grids
    rf_params = config['RandomForest']
    xgb_params = config['XGBoost']

    # Loop per route
    for route in tqdm(routes, desc='Tuning models by route'):   
        route_df = df[df[config['data']['route_col']] == route].sort_values(config['data']['time_col']).copy()
        
        X = route_df.drop(columns=[config['data']['features_to_drop']])
        y = route_df[config['data']['target']]

        # Random Forest
        with mlflow.start_run(run_name=f"{route}_RandomForest"):
            rf = RandomForestRegressor(random_state=config["search"]["random_state"])
            rf_search = RandomizedSearchCV(
                rf,
                rf_params,
                n_iter=config["search"]["n_iter"],
                cv=tscv,
                scoring=config["search"]["scoring"],
                verbose=1,
                random_state=config["search"]["random_state"],
                n_jobs=config["search"]["n_jobs"]
            )
            rf_search.fit(X, y)

            best_rf = rf_search.best_estimator_
            mlflow.log_params(rf_search.best_params_)
            mlflow.log_metric("best_cv_mae", -rf_search.best_score_)
            mlflow.sklearn.log_model(best_rf, artifact_path=f"{route}_rf_best")
            joblib.dump(best_rf, os.path.join(SAVE_DIR, f"{route}_rf_best.pkl"))


        # XGBoost
        with mlflow.start_run(run_name=f"{route}_XGBoost"):
            xgb = XGBRegressor(random_state=config["search"]["random_state"])
            xgb_search = RandomizedSearchCV(
                xgb,
                xgb_params,
                n_iter=config["search"]["n_iter"],
                cv=tscv,
                scoring=config["search"]["scoring"],
                verbose=1,
                random_state=config["search"]["random_state"],
                n_jobs=config["search"]["n_jobs"]
            )
            xgb_search.fit(X, y)

            best_xgb = xgb_search.best_estimator_
            mlflow.log_params(xgb_search.best_params_)
            mlflow.log_metric("best_cv_mae", -xgb_search.best_score_)
            mlflow.sklearn.log_model(best_xgb, artifact_path=f"{route}_xgb_best")
            joblib.dump(best_xgb, os.path.join(SAVE_DIR, f"{route}_xgb_best.pkl"))

        

if __name__ == "__main__":
    main()

