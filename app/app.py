import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="AMI Mortality Prediction Dashboard", layout="wide")

@st.cache_resource
def load_models():
    # Load XGBoost Models
    with open('models/enhanced_xgb_model.pkl', 'rb') as f:
        enhanced_model = pickle.load(f)
    with open('models/baseline_xgb_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
        
    # Load the Scaler
    with open('models/vitals_scaler.pkl', 'rb') as f:
        vitals_scaler = pickle.load(f)
        
    # NLP Model 
    local_model_path = "../results/models/finetuned_tiny_clinicalbert" 
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        nlp_model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=2)
        nlp_model.eval() 
    except Exception as e:
        st.error(f"Failed to load BERT: {e}")
        tokenizer, nlp_model = None, None 
    
    return enhanced_model, baseline_model, tokenizer, nlp_model, vitals_scaler

enhanced_xgb, baseline_xgb, tokenizer, nlp_model, vitals_scaler = load_models()

def clean_clinical_text(text):
    """
    Cleans raw MIMIC-III discharge summaries for NLP processing.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.replace('___', '')
    
    text = re.sub(r'Name:.*?Unit No:.*?\n', '', text)
    text = re.sub(r'Admission Date:.*?Discharge Date:.*?\n', '', text)
    text = re.sub(r'Date of Birth:.*?Sex:.*?\n', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_nlp_risk_score(text):
    """Passes the TAIL of the text through TinyClinicalBERT."""
    if tokenizer is None or nlp_model is None:
        return 0.5 
    
    cleaned_text = clean_clinical_text(text)
    
    tail_text = cleaned_text[-2500:]
    
    inputs = tokenizer(tail_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    
    with torch.no_grad():
        outputs = nlp_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return float(probabilities[0][1].item())

EXPECTED_FEATURES = [
    'admission_age', 'Troponin_T_max', 'CK_MB_max', 
    'gender_F', 'gender_M', 
    'admission_type_AMBULATORY OBSERVATION', 'admission_type_DIRECT EMER.', 
    'admission_type_DIRECT OBSERVATION', 'admission_type_ELECTIVE', 
    'admission_type_EU OBSERVATION', 'admission_type_EW EMER.', 
    'admission_type_OBSERVATION ADMIT', 'admission_type_SURGICAL SAME DAY ADMISSION', 
    'admission_type_URGENT', 'insurance_Medicaid', 'insurance_Medicare', 
    'insurance_No charge', 'insurance_Other', 'insurance_Private',
    'nlp_finetuned_risk_score' 
]

st.sidebar.header("Input Patient Data")

age = st.sidebar.slider("Admission Age", 18, 100, 65)
gender = st.sidebar.selectbox("Gender", ["M", "F"])

adm_options = [
    'EW EMER.', 'DIRECT EMER.', 'URGENT', 'ELECTIVE', 
    'OBSERVATION ADMIT', 'AMBULATORY OBSERVATION', 
    'DIRECT OBSERVATION', 'EU OBSERVATION', 'SURGICAL SAME DAY ADMISSION'
]
adm_type = st.sidebar.selectbox("Admission Type", adm_options)

ins_options = ['Medicare', 'Private', 'Medicaid', 'Other', 'No charge']
insurance = st.sidebar.selectbox("Insurance Type", ins_options)

st.sidebar.subheader("Lab Results")
trop_t = st.sidebar.number_input("Troponin T max", value=0.05, format="%.3f")
ck_mb = st.sidebar.number_input("CK-MB max", value=5.0, format="%.1f")

st.sidebar.subheader("Clinical Note")
clinical_note = st.sidebar.text_area("Nursing/Attending Notes:", height=150, 
    value="Patient presents with severe chest pain. Pallor and diaphoresis noted. High risk of decompensation.")


st.title("ICU Triage Dashboard: AMI Mortality Prediction")

if st.sidebar.button("Run Risk Analysis"):
    with st.spinner("Analyzing physiological data and clinical text..."):
        
        # 1. LIVE BERT INFERENCE
        bert_risk_score = get_nlp_risk_score(clinical_note)
        
        # Scale the raw numerical inputs
        raw_vitals = [[age, trop_t, ck_mb]]
        scaled_vitals = vitals_scaler.transform(raw_vitals)[0] 
        
        scaled_age = scaled_vitals[0]
        scaled_trop = scaled_vitals[1]
        scaled_ckmb = scaled_vitals[2]
        
        # 2. BULLETPROOF DATAFRAME CONSTRUCTION
        data_dict = {feat: [0.0] for feat in EXPECTED_FEATURES}
        
        data_dict['admission_age'] = [float(scaled_age)]
        data_dict['Troponin_T_max'] = [float(scaled_trop)]
        data_dict['CK_MB_max'] = [float(scaled_ckmb)]
        data_dict['nlp_finetuned_risk_score'] = [float(bert_risk_score)] 
        
        data_dict[f'gender_{gender}'] = [1.0]
        data_dict[f'admission_type_{adm_type}'] = [1.0]
        data_dict[f'insurance_{insurance}'] = [1.0]
        
        input_data = pd.DataFrame(data_dict)[EXPECTED_FEATURES].astype(float)
        baseline_input = input_data.drop(columns=['nlp_finetuned_risk_score'])

        # 3. Predict Probabilities
        try:
            enhanced_prob = enhanced_xgb.predict_proba(input_data)[0][1]
            baseline_prob = baseline_xgb.predict_proba(baseline_input)[0][1]
            
            # 5. Display Results
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Baseline Model (Vitals Only)")
                st.metric("Predicted Mortality Risk", f"{baseline_prob*100:.1f}%")
                
            with col2:
                st.error("Enhanced Multimodal Model")
                delta_val = (enhanced_prob - baseline_prob) * 100
                st.metric("Predicted Mortality Risk", f"{enhanced_prob*100:.1f}%", f"{delta_val:+.1f}% from NLP")
                st.caption(f"**Live BERT Risk Score:** {bert_risk_score:.3f}")

            # 6. SHAP EXPLAINABILITY 
            st.subheader("AI Decision Explainability (SHAP)")
            
            # 1. Create a neutral baseline (all zeros) to bypass the XGBoost JSON bug
            background = pd.DataFrame(np.zeros((1, len(EXPECTED_FEATURES))), columns=EXPECTED_FEATURES)

            # 2. Wrapper function: CRITICAL FIX -> Return ONLY Class 1 (Mortality) probabilities
            def predict_fn(X_array):
                df = pd.DataFrame(X_array, columns=EXPECTED_FEATURES)
                return enhanced_xgb.predict_proba(df)[:, 1]  # <--- The [:, 1] is the magic fix!

            # 3. Initialize KernelExplainer
            explainer = shap.KernelExplainer(predict_fn, background)

            # 4. Calculate SHAP values (silent=True to prevent terminal spam)
            shap_values_obj = explainer.shap_values(input_data, silent=True)

            # 5. Extract safely (Now guaranteed to be single-output)
            base_val = explainer.expected_value
            
            # Safely unwrap the base value if it's inside a single-item array
            if isinstance(base_val, (np.ndarray, list)):
                base_val = base_val[0]

            # 6. Reconstruct the Explanation object cleanly
            clean_explanation = shap.Explanation(
                values=np.array(shap_values_obj[0], dtype=float),
                base_values=float(base_val),
                data=np.array(input_data.iloc[0], dtype=float),
                feature_names=EXPECTED_FEATURES
            )

            # 7. Draw the Waterfall Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(clean_explanation, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.write("Check your model files or console logs for details.")

else:
    st.info("👈 Enter data in the sidebar and click 'Run Risk Analysis'.")