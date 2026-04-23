# 🚗 Banff Traffic Management System

---

## Project Overview

The **Banff Traffic Management System** is a data-driven web application developed by **Team Alpine Analysts** as part of the *Machine Learning Analyst Diploma* at **NorQuest College (Fall 2025)**.  
The project applies **machine learning and data visualization** to analyze and forecast **traffic congestion** and **parking availability** across Banff National Park.

The system integrates multiple predictive and analytical modules into a single Streamlit interface, transforming real transportation data into actionable insights for smarter mobility management.

---

## Motivation

Banff is one of Canada’s most visited destinations, attracting millions of visitors every year.  
As tourism grows, the region faces significant challenges in **traffic congestion**, **parking scarcity**, and **visitor flow management**.

The goal of this project is to create a **decision-support platform** that helps stakeholders:
- Monitor and forecast real-time route conditions.
- Anticipate congestion patterns before they occur.
- Analyze parking demand, usage, and revenue trends.
- Support sustainable mobility through data-driven insights.

---

## Application Pages

The Streamlit dashboard consists of **four main interactive pages**, each focused on a distinct analysis area:

| Page | Description |
|------|--------------|
| **1. Traffic Delay Predictor** | Machine learning–powered model that estimates congestion probability and predicts per-route delay durations. |
| **2. Traffic Analysis Dashboard** | Interactive visualizations of historical speed, volume, and delay trends across Banff’s main routes. |
| **3. Parking Forecast** | Predicts parking lot occupancy 60 minutes into the future using a regression-based XGBoost model trained on 15-minute intervals. |
| **4. Parking Analytics Dashboard** | Aggregates parking session data to visualize total sessions, revenue, duration distributions, and peak demand by hour, day, and month. |

Each page is designed for usability, transparency, and consistency — maintaining the same layout, typography, and visual structure.

---

## Technical Overview

The Banff Traffic Management System combines predictive modeling with interactive visualization.  
It was developed entirely in **Python**, using:

- **Machine Learning:** `scikit-learn`, `XGBoost`
- **Data Processing:** `pandas`, `NumPy`
- **Visualization:** `Plotly`, `Streamlit`
- **Data Storage:** `Google Drive` (private data hosting)
- **Deployment:** `Streamlit Cloud`

### Core Models
- **Delay Risk Classifier** – Predicts whether a route will experience *no delay*, *minor delay*, or *major congestion*.
- **Per-Route Delay Regressor** – Forecasts the expected delay duration (in minutes) per route using temporal and lag-based features.
- **Parking Occupancy Forecaster** – Uses a regression-based XGBoost model to predict 60-minute-ahead lot occupancy.

All models and dashboards are integrated into one unified **Streamlit application** providing dynamic visualization, caching, and seamless navigation.

---

## Machine Learning Workflow

The project followed the **CRISP-DM** methodology for both traffic and parking forecasting components:

1. **Data Understanding** – Explored over a year of traffic and parking data from Banff National Park, including speed, travel time, delay, occupancy, and volume metrics.  
2. **Data Preparation** – Cleaned and merged datasets, engineered lag and rolling-window features, and derived temporal variables such as hour, day of week, and seasonality.  
3. **Modeling** –  
   - **Traffic Models:** Trained Random Forest and XGBoost regressors to predict route-specific delays and congestion levels.  
   - **Parking Models:** Developed XGBoost regressors to forecast 60-minute occupancy based on recent usage and time-based trends.  
4. **Evaluation** – Measured model performance using **MAE**, **RMSE**, and **R²** for regression tasks, and **precision/recall** for classification benchmarks.  
5. **Deployment** – Deployed all trained models into a unified **Streamlit** dashboard for real-time prediction, visualization, and data exploration.

---

## Technology Stack

| Category | Tools |
|-----------|-------|
| **Languages** | Python |
| **Libraries** | pandas, scikit-learn, XGBoost, Random Forest, Plotly, Streamlit |
| **Data Storage** | Google Drive (secure private access) |
| **Deployment** | Streamlit Cloud |
| **Visualization** | Streamlit, Plotly Express |
| **Workflow** | CRISP-DM & Agile |

---

## Page Summaries

### 1. **Traffic Delay Predictor**
- Predicts congestion levels and delay durations per route.  
- Displays risk categories with color-coded confidence levels.  
- Allows users to explore both global and route-specific model outputs.

### 2. **Traffic Analysis Dashboard**
- Provides visual summaries of historical delay data.  
- Includes interactive charts showing daily, weekly, and monthly delay trends.  
- Supports pattern discovery across routes and time periods.

### 3. **Parking Forecast**
- Predicts occupancy levels 60 minutes ahead using recent historical data.  
- Displays metrics for **current occupancy**, **predicted occupancy**, and **available spaces**.  
- Uses an XGBoost regression model trained on 15-minute parking data.  
- Updated layout mirrors the Traffic Predictor for visual consistency.

### 4. **Parking Analytics Dashboard**
- Aggregates session data from all Banff parking units.  
- Displays KPIs such as total sessions, total revenue, and average duration.  
- Includes interactive charts for:
  - Monthly session and revenue trends  
  - Day-of-week activity  
  - Payment type distribution  
  - Duration and heatmap visualizations  
- Optimized with top-row filters for performance and usability.

---

## Deployment

The application is deployed using **Streamlit Cloud** with secure access to data and models hosted privately on Google Drive.  
Each component loads models, data, and visual assets dynamically to ensure modular updates and maintain privacy.

---

## Conclusion

The **Banff Traffic Management System** demonstrates how machine learning can enhance transportation planning and operational efficiency in a national park setting.  
By combining predictive analytics with interactive visualization, this project offers a scalable foundation for data-informed decision-making across transportation systems.

---

## Team Alpine Analysts

| Name | Role |
|------|------|
| **Angela Lekivetz** | Data Analysis · Model Development · Streamlit Integration |
| **Christine Joyce Moraleja** | Data Cleaning · Documentation · Project Coordination |
| **Victoriia Biaragova** | Model Development · Feature Engineering · Dashboard Design |
| **Sirjana Chauhan** | Model Evaluation · Visualization · Testing |

---

## Acknowledgment

Developed as part of **CMPT 3835 – Work Integrated Learning 2**  
**NorQuest College · Fall 2025**  
**Instructors:** Uchenna Mgbaja · Palwasha Afsar

---

