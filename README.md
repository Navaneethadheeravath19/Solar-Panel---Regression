![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

# ☀️ Solar Panel Power Prediction — Regression Model

An end-to-end machine learning project that predicts solar panel power output using environmental data such as distance to solar noon, temperature, humidity, wind speed, and sky cover.

## 📌 Project Overview
This project builds and evaluates regression models to forecast solar power generation based on real-world environmental factors. The goal is to support short-term energy forecasting and improve solar energy planning.

## 🎯 Objective
- Predict solar power output using environmental features
- Compare multiple regression models for best accuracy
- Handle skewed targets and outliers for better model performance
- Build a deployment-ready prediction pipeline

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Linear Regression, Random Forest) |
| Evaluation | RMSE, MAE, R² Score |
| Notebook | Jupyter Notebook |

## 📁 Project Structure
```
Solar-Panel---Regression/
│
├── solar_power_generation_EDA.ipynb   # Main analysis & modeling notebook
├── deploy_app.py                       # Deployment script
├── solarpowergeneration.csv            # Dataset
└── README.md
```

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| distance-to-solar-noon | Distance from solar noon (key predictor) |
| temperature | Ambient temperature |
| wind-direction | Wind direction in degrees |
| wind-speed | Wind speed |
| sky-cover | Cloud/sky cover level |
| visibility | Visibility in miles |
| humidity | Relative humidity |
| average-wind-speed | Average wind speed over period |
| average-pressure | Atmospheric pressure |
| power-generated | **Target variable** — solar power output |

## 📊 Methodology
1. **Data Collection** — Real-world solar power generation dataset with 5000+ records
2. **EDA** — Distribution plots, correlation analysis, outlier detection
3. **Data Cleaning** — Handled skewed targets and outliers
4. **Feature Engineering** — Selected key environmental predictors
5. **Model Building** — Linear Regression and Random Forest Regressor
6. **Evaluation** — RMSE, MAE, and R² Score metrics
7. **Deployment** — Built deployment-ready prediction pipeline

## 📈 Key Results
- **Best Model:** Random Forest Regressor
- **Accuracy:** ~80% (R² Score)
- **Key Finding:** Distance to solar noon is the strongest predictor of power output

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/Navaneethadheeravath19/Solar-Panel---Regression.git
cd Solar-Panel---Regression

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Run the Jupyter Notebook
jupyter notebook solar_power_generation_EDA.ipynb
```

## 📂 Dataset
- **Source:** Real-world solar power generation data
- **Format:** .csv
- **Records:** 5000+
- **Features:** 9 environmental features + 1 target variable

## 🔮 Limitations & Future Work
- Could incorporate time-series forecasting (LSTM, ARIMA)
- Adding more weather features could improve accuracy
- Real-time prediction API can be built using FastAPI or Flask

## 👤 Author
**Navaneetha Dheeravath**
📧 navaneethadheeravath19@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/navaneetha19)
💻 [GitHub](https://github.com/Navaneethadheeravath19)

---
⭐ If you found this project useful, please give it a star!
