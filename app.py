"""Fraud Detection Dashboard"""
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import os
import numpy as np

st.set_page_config(page_title="Fraud Detection", layout="wide")

@st.cache_resource
def load_data():
    return pd.read_csv('creditcard.csv')

@st.cache_resource
def load_models():
    """Try to load pre-trained models if they exist"""
    models = {'lr': None, 'rf': None}
    
    if os.path.exists('models/logistic_regression.pkl'):
        with open('models/logistic_regression.pkl', 'rb') as f:
            models['lr'] = pickle.load(f)
    
    if os.path.exists('models/random_forest.pkl'):
        with open('models/random_forest.pkl', 'rb') as f:
            models['rf'] = pickle.load(f)
    
    return models

df = load_data()
models = load_models()

page = st.sidebar.radio("Pages", ["Dashboard", "Performance", "Dataset", "Live Prediction", "Report"])

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

elif page == "Live Prediction":
    st.title("🔮 Live Fraud Detection")
    st.markdown("---")
    
    # Sample data
    legit_data = "0, -1.35980713, -0.0727811733, 2.53634674, 1.37815522, -0.33832077, 0.462387778, 0.239598554, 0.0986979013, 0.36378697, 0.090794172, -0.551599533, -0.617800856, -0.991389847, -0.311169354, 1.46817697, -0.470400525, 0.207971242, 0.0257905802, 0.40399296, 0.251412098, -0.0183067779, 0.277837576, -0.11047391, 0.0669280749, 0.128539358, -0.189114844, 0.133558377, -0.0210530535, 149.62"
    fraud_data = "406, -2.31222654, 1.95199201, -1.60985073, 3.99790559, -0.522187865, -1.42654532, -2.53738731, 1.39165725, -2.77008928, -2.77227214, 3.20203321, -2.89990739, -0.595221881, -4.28925378, 0.38972412, -1.14074718, -2.83005567, -0.0168224682, 0.416955705, 0.126910559, 0.517232371, -0.0350493686, -0.465211076, 0.320198199, 0.0445191675, 0.177839798, 0.261145003, -0.143275875, 0.0"
    
    if models['rf'] is None and models['lr'] is None:
        st.warning("⚠️ Models not loaded. Run `python main.py` first.")
    else:
        model_choice = st.radio("Select Model", ["Logistic Regression", "Random Forest"], horizontal=True)
        selected_model = models['lr'] if model_choice == "Logistic Regression" else models['rf']
        
        # Sample data display
        col1, col2 = st.columns(2)
        with col1:
            st.info("**✅ LEGITIMATE Sample**\n\n`Click to copy and paste below`")
            st.code(legit_data, language="text")
        
        with col2:
            st.error("**🚨 FRAUDULENT Sample**\n\n`Click to copy and paste below`")
            st.code(fraud_data, language="text")
        
        # Automatically test sample data with SELECTED model
        st.markdown("---")
        st.subheader(f"📊 Sample Data Predictions ({model_choice})")
        
        try:
            legit_values = np.array([float(v.strip()) for v in legit_data.split(',')]).reshape(1, -1)
            fraud_values = np.array([float(v.strip()) for v in fraud_data.split(',')]).reshape(1, -1)
            
            # Legitimate prediction using SELECTED model
            legit_pred = selected_model.predict(legit_values)[0]
            legit_prob = selected_model.predict_proba(legit_values)[0]
            legit_conf = legit_prob[1] if legit_pred == 1 else legit_prob[0]
            
            # Fraud prediction using SELECTED model
            fraud_pred = selected_model.predict(fraud_values)[0]
            fraud_prob = selected_model.predict_proba(fraud_values)[0]
            fraud_conf = fraud_prob[1] if fraud_pred == 1 else fraud_prob[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if legit_pred == 1:
                    st.error(f"🚨 FRAUDULENT (Confidence: {legit_conf*100:.2f}%)")
                else:
                    st.success(f"✅ LEGITIMATE (Confidence: {legit_conf*100:.2f}%)")
            
            with col2:
                if fraud_pred == 1:
                    st.error(f"🚨 FRAUDULENT (Confidence: {fraud_conf*100:.2f}%)")
                else:
                    st.success(f"✅ LEGITIMATE (Confidence: {fraud_conf*100:.2f}%)")
        
        except Exception as e:
            st.warning(f"⚠️ Could not test sample data: {str(e)}")
        
        st.markdown("---")
        st.subheader("📋 Enter Transaction Features")
        user_input = st.text_area("30 comma-separated values (Time, V1-V28, Amount)", height=80, placeholder="e.g., 0, -1.35, -0.07, 2.53, ...")
        
        if st.button("🔍 Predict", type="primary", use_container_width=True):
            # Validation 1: Check if input is empty
            if not user_input or not user_input.strip():
                st.error("❌ Input field is empty. Please enter 30 comma-separated values.")
            else:
                try:
                    # Validation 2: Split and clean input
                    raw_values = user_input.split(',')
                    
                    if len(raw_values) == 0:
                        st.error("❌ No values found. Please enter comma-separated numbers.")
                    else:
                        # Validation 3: Convert to float and handle invalid values
                        values = []
                        invalid_indices = []
                        
                        for idx, v in enumerate(raw_values):
                            cleaned = v.strip()
                            if not cleaned:
                                invalid_indices.append(idx + 1)
                            else:
                                try:
                                    values.append(float(cleaned))
                                except ValueError:
                                    st.error(f"❌ Invalid value at position {idx + 1}: '{cleaned}' is not a number.")
                                    break
                        else:
                            # Validation 4: Check if we have the correct number of values
                            if len(values) != 30:
                                st.error(f"❌ Expected 30 values, got {len(values)}. Please check your input.")
                            else:
                                # Validation 5: Check for NaN or Inf values
                                values_array = np.array(values)
                                if np.isnan(values_array).any() or np.isinf(values_array).any():
                                    st.error("❌ Input contains NaN or Inf values. Please enter valid numbers.")
                                else:
                                    # All validations passed - make prediction with SELECTED model
                                    test_data = np.array(values).reshape(1, -1)
                                    prediction = selected_model.predict(test_data)[0]
                                    prob = selected_model.predict_proba(test_data)[0]
                                    confidence = prob[1] if prediction == 1 else prob[0]
                                    
                                    if prediction == 1:
                                        st.error("🚨 FRAUDULENT")
                                    else:
                                        st.success("✅ LEGITIMATE")
                                    
                                    st.metric("Confidence", f"{confidence*100:.2f}%")
                        
                except ValueError as e:
                    st.error(f"❌ Error parsing input: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

elif page == "Report":
    st.title("Report")
    with open("output/charts/index.html", 'r', encoding='utf-8') as f:
        st.components.v1.html(f.read(), height=1200, scrolling=True)

