import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go

# -----------------------------------------
# PAGE CONFIG 
# -----------------------------------------

st.set_page_config(page_title="Churn Predictor", layout ="wide")

st.title("Customer Churn Prediction")

# --------------------------------------------------
# LOAD DATA 
# ----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Hrishikesh/churn-mlops-platform/streamlit_app/Telco-Customer-Churn.csv")
    return df

df = load_data()

# --------------------------------------------------------------------
# SHOW DATA
# ----------------------------------------------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())


# ---------------------------------------------------------------------
# TARGET 
# ---------------------------------------------------------------------

target = "Churn"
X = df.drop(columns=[target, "customerID"])
y = df[target].map({"Yes":1, "No":0})

categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns


# -------------------------------------------------------------------
# PIPELINE  
# -------------------------------------------------------------------

preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                                 ("num", "passthrough", numerical_cols)
                                 ,])

model = GradientBoostingClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", model),
    ]
)

# -------------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)

st.success(f"Model Trained Successfully !! | ROC-AUC = {auc:.3f}")


# --------------------------------------------------
# SIDEBAR INPUT FORM 
# -------------------------------------------------------------

st.sidebar.header("Customer Details")

input_data = {}

for col in X.columns:
    if col in categorical_cols:
        input_data[col] = st.sidebar.selectbox(col, df[col].unique())
    else:
        input_data[col] = st.sidebar.number_input(
            col,
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].median())
        )

input_df = pd.DataFrame([input_data])

# --------------------------------------------------------------------
# PREDICT
# --------------------------------------------------------------------

if st.button("Predict Churn"):
    
    prediction_proba = pipeline.predict_proba(input_df)[0][1]
    

    st.subheader("Prediction Result")

    st.metric(label = "Churn Probability",
             value = f"{prediction_proba * 100:.2f}%"
             )
    
    if prediction_proba > 0.5:
        st.error("⚠️ High Risk Customer")
    else:
        st.success("✅ Customer Likely to Stay")

    # Bar chart
    fig, ax = plt.subplots()

    labels = ["Stay", "Churn"]
    values = [1 - prediction_proba, prediction_proba]

    sns.barplot(x = labels, y = values, ax = ax, palette="viridis")

    ax.set_ylabel("probability")
    ax.set_ylim(0,1)
    ax.set_title("Churn Probability Distribution")

    st.pyplot(fig)


# streamlit run "C:\Users\Hrishikesh\churn-mlops-platform\streamlit_app\app.py"












