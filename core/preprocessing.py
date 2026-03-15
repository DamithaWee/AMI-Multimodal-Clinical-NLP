import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

def load_and_merge_data(structured_path, nlp_features_path):
    """
    Merges structured clinical data with NLP-extracted features.
    """
    logging.info("Loading datasets...")
    df_struct = pd.read_csv(structured_path)
    df_nlp = pd.read_csv(nlp_features_path)
    merged_df = pd.merge(df_struct, df_nlp, on='hadm_id', how='inner')
    
    logging.info(f"Merged Data Shape: {merged_df.shape}")
    return merged_df


def preprocess_multimodal_data(df):
    """
    Cleans, imputes, and encodes the data for ML training.
    """
    # 1. Define Target Variable
    if 'hospital_expire_flag' in df.columns:
        y = df['hospital_expire_flag']
    else:
        raise ValueError("Target variable 'hospital_expire_flag' not found.")

    # 2. Define Feature Columns
    numeric_features = [
        'admission_age', 
        'Troponin_T_max', 
        'CK_MB_max', 
        'nlp_sentiment_score', 
        'nlp_uncertainty_score'
    ]
    
    # B. Categorical Features (Demographics)
    categorical_features = [
        'gender', 
        'admission_type', 
        'insurance'
    ]
    
    # 3. Define Preprocessing Pipelines
    
    # Pipeline for Numerical Data:
    # - Impute missing values (e.g., missing Troponin) with the Median
    # - Scale features to mean=0, var=1 (Crucial for Neural Networks/SVM)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for Categorical Data:
    # - One-Hot Encode (Convert 'M/F' to vectors)
    # - handle_unknown='ignore' prevents crashes on rare categories in test set
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine into a single preprocessor object
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 4. Apply Transformations
    logging.info("Preprocessing data (Imputation + Normalization)...")
    X = preprocessor.fit_transform(df)
    
    # Get feature names back for interpretability (SHAP) later
    try:
        cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(cat_feature_names)
    except:
        feature_names = numeric_features + [f"cat_{i}" for i in range(X.shape[1] - len(numeric_features))]

    logging.info("Preprocessing complete.")
    return X, y, feature_names