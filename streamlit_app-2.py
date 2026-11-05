import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

# Page config
st.set_page_config(page_title="Red Wine Quality Classifier", layout="wide")

# Advanced UI Inject: Vineyard Background & Custom Wine Glass Drag Handles
st.markdown("""
<style>
/* Fullscreen Vineyard background */
.stApp {
    background-image: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)), 
                      url("https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?q=80&w=1600&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #ffffff;
}

/* Text adjustments for visibility */
h1, h2, h3, h4, p, span, label, .stMarkdown {
    color: #ffffff !important;
}

/* Sidebar styling override */
[data-testid="stSidebar"] {
    background-color: rgba(43, 8, 18, 0.9) !important;
    border-right: 2px solid #58111A;
}

/* Customize Slider Handles into wine glass shape */
div[role="slider"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: 28px !important; 
    height: 28px !important;
    margin-top: -6px !important;
}
div[role="slider"]::before {
    content: "🍷" !important;
    font-size: 24px !important;
    line-height: 1 !important;
    display: block !important;
    cursor: grab !important;
}
div[role="slider"]:active::before {
    cursor: grabbing !important;
    transform: scale(1.2);
}

/* Tabs style adjustments */
button[data-baseweb="tab"] {
    color: #cccccc !important;
}
button[aria-selected="true"] {
    color: #ff4b4b !important;
    border-bottom-color: #ff4b4b !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🍷 Red Wine Quality Interactive Classifier")
st.write("Deploy this app instantly to Streamlit Cloud. It trains 5 machine learning models and performs metrics assessment, classification reports, and real-time predictions.")

# --- Step 1: Dataset Loading ---
st.sidebar.header("📁 Dataset Source")
uploaded_file = st.sidebar.file_uploader("Upload Wine Dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        # Automatically detect separator
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')

    # Convert target binary column if quality exists
    if 'quality' in df.columns and 'target' not in df.columns:
        df['target'] = (df['quality'] >= 6).astype(int)
    return df

try:
    df = load_data(uploaded_file)

    # Features (X) and Target (y)
    feature_cols = [col for col in df.columns if col not in ['quality', 'target']]
    X = df[feature_cols]
    y = df['target'] if 'target' in df.columns else df['quality']

    # Train-Test Split (test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.info(f"📊 **Dataset Summary**: {df.shape[0]} rows, {df.shape[1]} columns loaded")

    # Model Training and Metrics Evaluator
    @st.cache_resource
    def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "K-Nearest Neighbor": KNeighborsClassifier(),
            "Naive Bayes (Gaussian)": GaussianNB(),
            "Support Vector Machine": SVC(probability=True, random_state=42)
        }
        trained_models = {}
        results = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC Score": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "MCC Score": matthews_corrcoef(y_test, y_pred)
            }
            results.append(metrics)
        return trained_models, pd.DataFrame(results)

    trained_models, results_df = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Slider setup
    st.sidebar.header("🧪 Input Wine Features")
    input_data = {}
    for col in X.columns:
        min_val = float(X_test[col].min())
        max_val = float(X_test[col].max())
        mean_val = float(X_test[col].mean())
        input_data[col] = st.sidebar.slider(f"{col.title()}", min_val, max_val, mean_val)

    # Tab layouts
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "📈 Diagnostics (Confusion Matrix)", "🔮 Make a Prediction", "🔍 View Test Data"])

    with tab1:
        st.header("Overall Model Evaluation Metrics")
        # Use a deep wine-red accent color for maximums to maintain clear contrast with white text
        st.dataframe(results_df.style.highlight_max(subset=results_df.columns[1:], color="#6b1d2f"))

    with tab2:
        st.header("Model Diagnostics")
        selected_diag_model = st.selectbox("Select Model for Diagnostics:", list(trained_models.keys()), key="diag")

        diag_model = trained_models[selected_diag_model]
        y_pred_diag = diag_model.predict(X_test_scaled)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred_diag)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Good', 'Good'], yticklabels=['Not Good', 'Good'], ax=ax)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Classification Report")
            report_dict = classification_report(y_test, y_pred_diag, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            # Assign index column name to give it a title in the UI
            report_df.index.name = "Metric / Class"
            st.dataframe(report_df)

    with tab3:
        st.header("Real-Time Wine Quality Prediction")
        selected_model_name = st.selectbox("Choose a model for prediction:", list(trained_models.keys()), key="predict")

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        model = trained_models[selected_model_name]
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else 0.5

        st.markdown("---")
        if prediction == 1:
            st.success(f"### 🎉 Prediction: **GOOD QUALITY WINE** (Confidence: {probability:.2%})")
        else:
            st.error(f"### 🛑 Prediction: **NOT GOOD QUALITY WINE** (Confidence: {1 - probability:.2%})")

    with tab4:
        st.header("Explore Test Set Instances")
        test_display = X_test.copy()
        test_display['Actual Target'] = y_test
        st.dataframe(test_display)

except Exception as e:
    st.error(f"Error processing dataset: {e}")
