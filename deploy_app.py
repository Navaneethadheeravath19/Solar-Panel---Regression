import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

st.set_page_config(page_title="☀️ Solar Power Prediction", layout="wide", page_icon="☀️")

# ── HEADER ──
st.title("☀️ Solar Panel Power Prediction")
st.markdown("**Predicting solar power output using environmental data | By Navaneetha Dheeravath**")
st.markdown("---")

# ── LOAD DATASET AUTOMATICALLY ──
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Navaneethadheeravath19/Solar-Panel---Regression/main/solarpowergeneration.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
numeric_df = df.select_dtypes(include=['float64', 'int64'])

st.success(f"✅ Dataset loaded automatically — {df.shape[0]} records, {df.shape[1]} features")

# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs(["📌 Dataset", "📊 EDA", "⚙️ ML Models", "🔮 Predict"])

# ── TAB 1: DATASET ──
with tab1:
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    st.subheader("📘 Summary Statistics")
    st.dataframe(numeric_df.describe(), use_container_width=True)

# ── TAB 2: EDA ──
with tab2:
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        col_hist = st.selectbox("Select feature for histogram:", numeric_df.columns)
        fig_hist = px.histogram(df, x=col_hist, nbins=30, marginal="box",
                                title=f"Distribution of {col_hist}",
                                color_discrete_sequence=["#00d4ff"])
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        x_col = st.selectbox("X-axis (Scatter):", numeric_df.columns)
        y_col = st.selectbox("Y-axis (Scatter):", numeric_df.columns, index=len(numeric_df.columns)-1)
        fig_scatter = px.scatter(df, x=x_col, y=y_col,
                                 title=f"{x_col} vs {y_col}",
                                 color_discrete_sequence=["#00d4ff"])
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📌 Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax_corr, fmt=".2f")
    st.pyplot(fig_corr)

# ── TAB 3: ML MODELS ──
with tab3:
    st.subheader("⚙️ Machine Learning — Regression Models")

    df_ml = numeric_df.fillna(numeric_df.mean())
    target = st.selectbox("Select Target Variable", numeric_df.columns,
                          index=list(numeric_df.columns).index('power-generated') if 'power-generated' in numeric_df.columns else 0)

    X = df_ml.drop(columns=[target])
    y = df_ml[target]

    test_size_val = st.slider("Test Size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val/100, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.01),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        results[name] = {
            "R² Score": round(r2_score(y_test, preds), 4),
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4)
        }
        trained_models[name] = model

    results_df = pd.DataFrame(results).T
    st.dataframe(results_df, use_container_width=True)

    best_model_name = results_df["R² Score"].idxmax()
    st.success(f"🏆 Best Model: **{best_model_name}** with R² Score: **{results_df.loc[best_model_name, 'R² Score']}**")

    # Bar chart of R² scores
    fig_bar = px.bar(results_df.reset_index(), x="index", y="R² Score",
                     title="Model Comparison — R² Score",
                     color="R² Score", color_continuous_scale="blues",
                     labels={"index": "Model"})
    st.plotly_chart(fig_bar, use_container_width=True)

    # Store best model and scaler in session
    st.session_state['best_model'] = trained_models[best_model_name]
    st.session_state['scaler'] = scaler
    st.session_state['features'] = list(X.columns)
    st.session_state['target'] = target

# ── TAB 4: PREDICT ──
with tab4:
    st.subheader("🔮 Predict Solar Power Output")

    if 'best_model' not in st.session_state:
        st.warning("⚠️ Please run the ML Models tab first to train the model!")
    else:
        st.info(f"Using best model from ML tab. Enter environmental values below:")

        df_ml2 = numeric_df.fillna(numeric_df.mean())
        features = st.session_state['features']
        target2 = st.session_state['target']
        X2 = df_ml2.drop(columns=[target2])

        cols = st.columns(3)
        user_input = {}
        for i, col in enumerate(features):
            with cols[i % 3]:
                user_input[col] = st.number_input(f"{col}", value=float(X2[col].mean()), format="%.4f")

        if st.button("⚡ Predict Power Output", type="primary"):
            user_df = pd.DataFrame([user_input])
            scaled_input = st.session_state['scaler'].transform(user_df)
            pred = st.session_state['best_model'].predict(scaled_input)
            st.success(f"🌟 Predicted Power Output: **{pred[0]:.4f}**")
            st.balloons()

# ── FOOTER ──
st.markdown("---")
st.markdown("**Navaneetha Dheeravath** | [LinkedIn](https://linkedin.com/in/navaneetha19) | [GitHub](https://github.com/Navaneethadheeravath19) | [Portfolio](https://navaneethadheeravath19.github.io)")
