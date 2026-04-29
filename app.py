import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# ------------------------------
# App Config (must be first)
# ------------------------------
st.set_page_config(
    page_title="ChurnPredictor Pro",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS for Dark & Light themes
# ------------------------------
def load_css(theme):
    """Return CSS string based on theme."""
    if theme == "Dark":
        background = "#0E0E0E"       # black-ish
        secondary_bg = "#1A1A2E"    # dark purple tint
        text_color = "#FFFFFF"
        accent = "#BB86FC"           # purple accent
        border = "#333366"
        button_bg = accent
        button_text = "#000000"
        sidebar_bg = "#121212"
        sidebar_text = "#FFFFFF"
    else:
        background = "#FFFFFF"
        secondary_bg = "#F5F0FF"    # very light purple
        text_color = "#1E1E1E"
        accent = "#7D3C98"           # deeper purple
        border = "#D1C4E9"
        button_bg = accent
        button_text = "#FFFFFF"
        sidebar_bg = "#F8F6FC"
        sidebar_text = "#1E1E1E"

    return f"""
    <style>
    /* Root variables */
    :root {{
        --bg: {background};
        --secondary-bg: {secondary_bg};
        --text: {text_color};
        --accent: {accent};
        --border: {border};
        --button-bg: {button_bg};
        --button-text: {button_text};
    }}

    /* Main container */
    .stApp {{
        background-color: {background};
        color: {text_color};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
        color: {sidebar_text};
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        color: {sidebar_text} !important;
    }}

    /* All text elements */
    h1, h2, h3, h4, h5, h6, p, span, div:not(.stException) {{
        color: {text_color} !important;
    }}

    /* Buttons */
    .stButton>button {{
        background-color: {button_bg};
        color: {button_text};
        border: 2px solid {accent};
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {text_color}22;
        border-color: {accent};
    }}

    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {{
        background-color: {secondary_bg};
        color: {text_color};
        border: 1px solid {border};
    }}

    /* Dataframe */
    .stDataFrame {{
        border: 1px solid {border};
    }}

    /* Toggle switch (theme toggle) */
    .theme-toggle {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 10px 0;
    }}

    /* Project name in sidebar */
    .sidebar-title {{
        font-size: 28px;
        font-weight: 700;
        color: {accent};
        text-align: center;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }}
    </style>
    """

# Initialize theme state
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"   # default

# Sidebar: Theme toggle + brand
with st.sidebar:
    st.markdown(
        f"<div class='sidebar-title'>⚡ ChurnPredictor Pro</div>",
        unsafe_allow_html=True
    )
    # Toggle
    theme = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
    st.session_state.theme = "Dark" if theme else "Light"

# Inject the current theme's CSS
st.markdown(load_css(st.session_state.theme), unsafe_allow_html=True)

# ------------------------------
# Now import the pipeline functions
# ------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.data_exploration import explore_data
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features, add_engineered_features
from src.data_preprocessing import preprocess_data
from src.random_forest import train_random_forest
from src.train_xgboost import train_xgboost    # renamed file
from src.model_comparison import compare_models

# ------------------------------
# Paths
# ------------------------------
DATA_RAW = os.path.join("data", "telco_data.csv")
CLEANED = os.path.join("data", "cleaned_data.csv")
ENGINEERED = os.path.join("data", "engineered_data.csv")
PREPROCESSOR = os.path.join("data", "results", "preprocessor.pkl")
RF_MODEL = os.path.join("data", "results", "best_rf_model.pkl")
XGB_MODEL = os.path.join("data", "results", "best_xgb_model.pkl")

def load_model(model_type):
    path = RF_MODEL if model_type == "rf" else XGB_MODEL
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ------------------------------
# Navigation
# ------------------------------
page = st.sidebar.radio("📌 Navigate", [
    "🏠 Home",
    "📊 Data Exploration",
    "⚙️ Run Pipeline",
    "🤖 Train Models",
    "📈 Model Comparison",
    "🔮 Predict Churn"
])

# =============================================
# HOME PAGE
# =============================================
if page == "🏠 Home":
    st.title("Customer Churn Prediction")
    st.markdown("""
    **ChurnPredictor Pro** leverages Random Forest and XGBoost to predict customer churn.
    
    - 📂 Upload your raw CSV
    - ⚙️ Run the preprocessing pipeline
    - 🤖 Train models with one click
    - 🔮 Get individual or batch predictions

    Use the **sidebar** to navigate between steps.
    """)
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(RF_MODEL):
            st.success("Random Forest model ready ✅")
        else:
            st.warning("Random Forest not trained")
    with col2:
        if os.path.exists(XGB_MODEL):
            st.success("XGBoost model ready ✅")
        else:
            st.warning("XGBoost not trained")

# =============================================
# DATA EXPLORATION
# =============================================
elif page == "📊 Data Exploration":
    st.title("Data Exploration")
    uploaded = st.file_uploader("Upload telco CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df.to_csv(DATA_RAW, index=False)
    else:
        if os.path.exists(DATA_RAW):
            df = pd.read_csv(DATA_RAW)
        else:
            st.error("No data found. Please upload a CSV.")
            st.stop()
    # Drop problematic column for display
    display_df = df.drop('customerID', axis=1) if 'customerID' in df.columns else df
    st.subheader("Raw data sample")
    st.dataframe(display_df.head(100))
    st.subheader("Descriptive statistics")
    st.write(display_df.describe(include='all'))

    if st.button("Generate EDA plots"):
        figs = explore_data(df)
        for fig in figs:
            st.pyplot(fig)

# =============================================
# RUN PIPELINE
# =============================================
elif page == "⚙️ Run Pipeline":
    st.title("Preprocessing Pipeline")
    st.markdown("Run cleaning → feature engineering → preprocessing → train/test split")

    if st.button("Start Pipeline"):
        if not os.path.exists(DATA_RAW):
            st.error("Please upload or place telco_data.csv in the data/ folder first.")
            st.stop()
        with st.spinner("Cleaning data..."):
            clean_data(DATA_RAW, CLEANED)
        with st.spinner("Engineering features..."):
            engineer_features(CLEANED, ENGINEERED)
        with st.spinner("Preprocessing & splitting..."):
            X_train, X_test, y_train, y_test = preprocess_data(ENGINEERED, PREPROCESSOR)
        st.success("Pipeline complete!")
        st.write(f"X_train shape: {X_train.shape}")
        st.write(f"y_train distribution:\n{pd.Series(y_train).value_counts()}")
        st.subheader("Processed feature columns")
        # The processed data is a numpy array, so just show first few
        st.write(pd.DataFrame(X_train).head())

# =============================================
# TRAIN MODELS
# =============================================
elif page == "🤖 Train Models":
    st.title("Model Training")
    model_choice = st.selectbox("Select model", ["Random Forest", "XGBoost", "Both"])

    if st.button("Train"):
        X_train = pd.read_csv("data/X_train.csv").values
        y_train = pd.read_csv("data/y_train.csv").values.ravel()

        if model_choice in ["Random Forest", "Both"]:
            with st.spinner("Training Random Forest..."):
                model, imp_df = train_random_forest(X_train, y_train)
                joblib.dump(model, RF_MODEL)
                imp_df.to_csv("data/results/feature_importance_rf.csv", index=False)
            st.success("Random Forest saved")
            st.subheader("Top 10 Features")
            st.bar_chart(imp_df.set_index('feature').head(10))

        if model_choice in ["XGBoost", "Both"]:
            with st.spinner("Training XGBoost..."):
                model, imp_df = train_xgboost(X_train, y_train)
                joblib.dump(model, XGB_MODEL)
                imp_df.to_csv("data/results/feature_importance_xg.csv", index=False)
            st.success("XGBoost saved")
            st.subheader("Top 10 Features")
            st.bar_chart(imp_df.set_index('feature').head(10))

# =============================================
# MODEL COMPARISON
# =============================================
elif page == "📈 Model Comparison":
    st.title("Model Comparison")
    try:
        X_test = pd.read_csv("data/X_test.csv").values
        y_test = pd.read_csv("data/y_test.csv").values.ravel()
    except FileNotFoundError:
        st.error("No test data found. Run the pipeline first.")
        st.stop()
    rf_model = load_model("rf")
    xgb_model = load_model("xg")
    if not rf_model or not xgb_model:
        st.error("Both models must be trained first.")
        st.stop()

    if st.button("Run Comparison"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Random Forest**")
            pred_rf = rf_model.predict(X_test)
            st.text(classification_report(y_test, pred_rf))
        with col2:
            st.markdown("**XGBoost**")
            pred_xgb = xgb_model.predict(X_test)
            st.text(classification_report(y_test, pred_xgb))

        st.subheader("ROC Curves")
        fig, ax = plt.subplots()
        for name, model in [("RF", rf_model), ("XGB", xgb_model)]:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Confusion Matrices")
        fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, (name, pred) in enumerate([("RF", pred_rf), ("XGB", pred_xgb)]):
            cm = confusion_matrix(y_test, pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Purples')
            axes[i].set_title(name)
        st.pyplot(fig2)

# =============================================
# PREDICT CHURN
# =============================================
elif page == "🔮 Predict Churn":
    st.title("Churn Prediction")
    model_type = st.radio("Select model", ["Random Forest", "XGBoost"])
    model = load_model("rf" if model_type == "Random Forest" else "xg")
    if model is None:
        st.error(f"{model_type} model not found. Train it first.")
        st.stop()
    if not os.path.exists(PREPROCESSOR):
        st.error("Preprocessor not found. Run the pipeline first.")
        st.stop()
    preprocessor = joblib.load(PREPROCESSOR)

    st.subheader("Single Customer Prediction")
    with st.form("single_pred"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", 0, 100, 1)
        with col2:
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        with col3:
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
            TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

        submitted = st.form_submit_button("Predict")
        if submitted:
            input_dict = {
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges
            }
            input_df = pd.DataFrame([input_dict])
            # Feature engineering
            input_df = add_engineered_features(input_df)
            processed = preprocessor.transform(input_df)
            pred = model.predict(processed)[0]
            proba = model.predict_proba(processed)[0][1]
            st.success(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
            st.metric("Churn Probability", f"{proba:.2%}")

    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload CSV with same columns as original data", type="csv")
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        batch_df = add_engineered_features(batch_df)   # Apply same engineering
        st.dataframe(batch_df.head())
        if st.button("Predict Batch"):
            processed = preprocessor.transform(batch_df)
            batch_df["Churn_Prediction"] = model.predict(processed)
            batch_df["Churn_Probability"] = model.predict_proba(processed)[:, 1]
            st.dataframe(batch_df)
            csv = batch_df.to_csv(index=False).encode()
            st.download_button("Download Predictions", csv, "predictions.csv", mime="text/csv")