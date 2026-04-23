# Models

Trained model files are not committed to this repository due to file size constraints.

All serialized models (`.joblib`, `.pkl`) are stored privately on Google Drive and loaded at runtime via GCP-authenticated access.

## Models

| Model | Type | Description |
|-------|------|-------------|
| Delay Risk Classifier | XGBoost | Classifies route congestion as no delay, minor delay, or major congestion |
| Per-Route Delay Regressor | Random Forest | Predicts expected delay duration (minutes) per route |
| Parking Occupancy Forecaster | XGBoost | Predicts 60-minute-ahead lot occupancy |

## Access

To run the application locally, you will need:
1. A valid GCP service account with access to the private Google Drive folder
2. Credentials configured in `streamlit/.streamlit/secrets.toml`

Contact the project team for access details.