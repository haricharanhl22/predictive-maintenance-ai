# Predictive Maintenance AI

> Predict machine failures before they happen using factory sensor data

ML system that monitors industrial equipment in real-time and predicts whether a machine is healthy, needs maintenance, or is about to fail — based on sensor readings (temperature, vibration, pressure, RPM, humidity).

## Demo

Move the sliders to simulate different sensor conditions and watch the AI predict machine health in real-time. Click **Run Simulation** to watch a machine deteriorate from healthy → warning → critical failure.

## How It Works
```
Sensor Data → StandardScaler → Random Forest → Health Prediction
(temp, vibration,    (normalize      (100 trees,      (healthy /
 pressure, RPM,       features)       balanced         warning /
 humidity)                            classes)         failure)
```

## Features

- **Real-time prediction** — adjust sliders, get instant health score
- **Risk percentages** — probability of healthy / warning / critical
- **Deterioration simulation** — watch status change over 100 steps
- **Feature importance** — vibration (34%) > RPM (32%) > temperature (22%)
- **Color-coded alerts** — green / yellow / red status indicators

## Tech Stack

- **scikit-learn** — Random Forest classifier, StandardScaler
- **pandas + numpy** — data generation and manipulation
- **joblib** — model serialization
- **Streamlit** — interactive dashboard with sliders
- **Python** — core language

## Model Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Healthy | ~0.97 | ~0.98 | ~0.97 |
| Warning | ~0.91 | ~0.88 | ~0.89 |
| Failure | ~0.94 | ~0.92 | ~0.93 |

## Feature Importance

| Sensor | Importance |
|--------|-----------|
| Vibration | 34.2% |
| RPM | 32.3% |
| Temperature | 22.1% |
| Pressure | 10.7% |
| Humidity | 0.7% |

Vibration and RPM are the strongest early failure indicators — consistent with real industrial maintenance knowledge.

## Setup
```bash
git clone https://github.com/haricharanhl22/predictive-maintenance-ai
cd predictive-maintenance-ai
pip install scikit-learn pandas numpy streamlit joblib
```

## Usage
```bash
# Generate dataset and train model
python generate_data.py
python train_model.py

# Launch dashboard
streamlit run app.py
```

## Dataset

Synthetically generated factory sensor data:
- **2000 samples** — 1500 healthy, 300 warning, 200 failure
- **5 features** — temperature, vibration, pressure, RPM, humidity
- **Realistic distributions** — normal operating ranges with failure anomalies

## Industry Relevance

Predictive maintenance is a core Industry 4.0 use case at companies like Bosch, Siemens, ABB, and Schaeffler. Traditional approaches use either reactive maintenance (fix when broken) or scheduled maintenance (fix on a calendar). Predictive maintenance uses real sensor data to intervene exactly when needed — reducing downtime costs significantly.

## Author

**Hari Charan Hosakote Lokesh**
- GitHub: [@haricharanhl22](https://github.com/haricharanhl22)
- LinkedIn: [haricharanhl22](https://linkedin.com/in/haricharanhl22)
