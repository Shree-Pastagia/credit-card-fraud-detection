"""Fraud Detection Dashboard"""
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Fraud Detection", layout="wide")

@st.cache_resource
def load_data():
    return pd.read_csv('creditcard.csv')

df = load_data()

page = st.sidebar.radio("Pages", ["Dashboard", "Performance", "Dataset", "Report"])

if page == "Dashboard":
    st.title("Fraud Detection Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", f"{len(df):,}")
    col2.metric("Fraud", f"{df[df['Class']==1].shape[0]:,}")
    col3.metric("Normal", f"{df[df['Class']==0].shape[0]:,}")
    
    col1, col2 = st.columns(2)
    col1.image(Image.open("output/charts/class_distribution.png"))
    col2.image(Image.open("output/charts/accuracy_comparison.png"))
    
    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [0.9994, 0.9997],
        'Precision': [0.9993, 0.9997],
        'Recall': [0.9994, 0.9997]
    }), use_container_width=True)

elif page == "Performance":
    st.title("Model Performance")
    col1, col2 = st.columns(2)
    col1.image(Image.open("output/charts/confusion_matrix_logistic_regression.png"))
    col2.image(Image.open("output/charts/confusion_matrix_random_forest.png"))
    st.image(Image.open("output/charts/metrics_comparison.png"))

elif page == "Dataset":
    st.title("Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    
    n = st.slider("Rows", 5, 100, 10)
    st.dataframe(df.head(n), use_container_width=True)
    
    if st.checkbox("Stats"):
        st.dataframe(df.describe(), use_container_width=True)

elif page == "Report":
    st.title("Report")
    with open("output/charts/index.html", 'r', encoding='utf-8') as f:
        st.components.v1.html(f.read(), height=1200, scrolling=True)

