import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import logging

class ClinicalNLP_Pipeline:
    def __init__(self):
        # 1. Detect GPU
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.device = torch.device(f"cuda:{self.device_id}") if self.device_id >= 0 else torch.device("cpu")
        
        logging.info(f"NLP Pipeline Initialized on: {self.device} (ID: {self.device_id})")
        
        # 2. Load Uncertainty Model
        self.unc_model_name = "bvanaken/clinical-assertion-negation-bert"
        self.tokenizer_unc = AutoTokenizer.from_pretrained(self.unc_model_name)
        self.model_unc = AutoModelForSequenceClassification.from_pretrained(self.unc_model_name)
        
        # CRITICAL FIX: Move the model to the GPU
        self.model_unc.to(self.device)
        self.model_unc.eval() # Set to evaluation mode (faster, no training gradients)
        
        # 3. Load Sentiment Model
        # The 'pipeline' function handles device movement automatically if 'device' arg is passed
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            device=self.device_id, # Must pass integer ID here, not device object
            top_k=None
        )

    def get_uncertainty_score(self, text):
        """
        Returns a score representing the 'Uncertainty Density' of the note.
        """
        if not isinstance(text, str) or len(text.strip()) < 5:
            return 0.0
            
        # Tokenize
        inputs = self.tokenizer_unc(text, return_tensors="pt", truncation=True, max_length=512)
        
        # CRITICAL FIX: Move input tensors to the GPU
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model_unc(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Index 2 = 'POSSIBLE' (Uncertainty)
        uncertainty_score = probs[0][2].item()
        return uncertainty_score

    def get_sentiment_score(self, text):
        """
        Returns a score (0.0 to 1.0) for NEGATIVE sentiment.
        """
        if not isinstance(text, str) or len(text.strip()) < 5:
            return 0.0
        
        try:
            # Pipeline handles GPU automatically because we passed device_id in __init__
            results = self.sentiment_pipe(text[:1024], truncation=True)
            for res in results[0]:
                if res['label'] == 'NEGATIVE':
                    return res['score']
        except Exception as e:
            logging.error(f"Error in sentiment: {e}")
            return 0.0
        
        return 0.0

    def process_dataframe(self, df, text_col='text'):
        tqdm.pandas(desc="Extracting NLP Features")
        df[text_col] = df[text_col].fillna("")
        
        # Note: .progress_apply is still a Python loop, but the GPU inference inside is fast.
        # For massive datasets (100k+ rows), we would use batch processing, 
        # but for this IRP, row-by-row on GPU is sufficient and safer to debug.
        
        logging.info(f"Extracting features from {len(df)} notes on {self.device}...")
        df['nlp_uncertainty_score'] = df[text_col].progress_apply(self.get_uncertainty_score)
        df['nlp_sentiment_score'] = df[text_col].progress_apply(self.get_sentiment_score)
        
        return df