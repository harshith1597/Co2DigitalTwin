# CO₂ Digital Twin — India

State-level CO₂ emission forecasting using a CatBoost ML model with an interactive Digital Twin dashboard built in Streamlit.

## Features
- 🗺️ India map with real-time Green/Yellow/Red CO₂ classification
- 🎚️ Sliders for Renewable Energy %, Energy Use, and Forecast Year
- 📊 Sensitivity analysis — see how RE% changes CO₂
- 📈 Historical trend + scenario comparison
- ⚡ Real CatBoost model predictions (R² = 0.9973)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
digital_twin/
├── app.py                        ← main Streamlit app
├── requirements.txt
├── data/
│   └── master_dataset_final.csv  ← 28 states, 1980–2024
└── models/
    ├── best_model_catboost.pkl
    ├── label_encoder_state.pkl
    ├── label_encoder_region.pkl
    └── scaler_X.pkl
```

## Model Performance

| Metric | Score |
|--------|-------|
| RMSE   | 3.66  |
| MAE    | 2.58  |
| MAPE   | 2.07% |
| R²     | 0.9973 |

## Dataset
- 28 Indian states, years 1980–2024
- Features: Energy use, RE generation, GDP, Urbanization
- Target: Carbon emissions (MtCO₂)
