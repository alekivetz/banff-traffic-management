"""
Evaluation pipeline for the Banff Route-Level Traffic Management project.
Loads trained models, evaluates holdout performance, and generates summary metrics and plots.

Supports a --config flag for consistent YAML-based execution.
"""

# --- Imports ---
import argparse
import logging
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Load config ---
def load_config(config_path: str = 'configs/evaluate_config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# --- Evaluate metrics ---
def evaluate_metrics(df: pd.DataFrame, target_col: str, pred_col: str = 'predicted') -> dict:
    y_true = df[target_col]
    y_pred = df[pred_col]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# --- Evaluate a single model ---
def evaluate_model(model_path: str, data_path: str, target_col: str, save_dir: str) -> dict:
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f'Loading model from {model_path}')
    model = joblib.load(model_path)

    logger.info(f'Loading evaluation data from {data_path}')
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preds = model.predict(X)
    df['predicted'] = preds

    metrics = evaluate_metrics(df, target_col, 'predicted')

    # Save predictions
    pred_path = os.path.join(save_dir, f'{os.path.basename(model_path).replace(".pkl", "_predictions.parquet")}')
    df.to_parquet(pred_path, index=False)
    logger.info(f'Saved predictions to {pred_path}')

    # Save metrics
    metrics_path = os.path.join(save_dir, f'{os.path.basename(model_path).replace(".pkl", "_metrics.json")}')
    pd.Series(metrics).to_json(metrics_path)
    logger.info(f'Saved metrics to {metrics_path}')

    return metrics


# --- Evaluate all models ---
def evaluate_all_models(model_dir: str, data_path: str, target_col: str, output_dir: str, mlflow_logging: bool = True):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    all_metrics = []

    if not model_files:
        logger.warning(f'No model files found in {model_dir}')
        return

    logger.info(f'Evaluating {len(model_files)} models found in {model_dir}')

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        metrics = evaluate_model(model_path, data_path, target_col, output_dir)

        if mlflow_logging:
            with mlflow.start_run(run_name=f'Evaluate_{os.path.basename(model_file)}', nested=True):
                mlflow.log_params({'model_file': model_file})
                mlflow.log_metrics(metrics)

        metrics['Model'] = model_file
        all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
    results_df.to_csv(summary_path, index=False)
    logger.info(f'Saved evaluation summary to {summary_path}')
    return results_df


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='Evaluate Banff Traffic Management models.')
    parser.add_argument('--config', type=str, default='configs/evaluate_config.yaml', help='Path to evaluation config YAML file.')
    parser.add_argument('--model-dir', type=str, help='Override model directory.')
    parser.add_argument('--data-path', type=str, help='Override evaluation dataset path.')
    parser.add_argument('--target-col', type=str, help='Override target column.')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow logging.')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    model_dir = args.model_dir or config['data']['model_dir']
    data_path = args.data_path or config['data']['eval_path']
    target_col = args.target_col or config['data']['target_col']
    output_dir = config['options']['output_dir']
    mlflow_logging = config['options']['mlflow_logging'] and not args.no_mlflow

    # Setup MLflow
    if mlflow_logging:
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])

    logger.info('Starting model evaluation...')
    evaluate_all_models(model_dir, data_path, target_col, output_dir, mlflow_logging)
    logger.info('✅ Evaluation complete.')


if __name__ == '__main__':
    main()
