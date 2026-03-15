<<<<<<< HEAD
<<<<<<< HEAD
# Multimodal EHR Fusion and Explainable AI in Medicine: Predictive Modeling of Mortality Risk

## Abstract
This repository contains the codebase and experimental framework for an undergraduate research project investigating the integration of unstructured clinical notes with structured Electronic Health Record (EHR) data. The primary objective is to improve the prediction of patient mortality risk in cohorts exhibiting extreme class imbalance, specifically addressing the 1:20 (approximately 5%) mortality distribution observed in the MIMIC-III database. By employing a Sequential/Late Fusion methodology, this study demonstrates the significant incremental gain achievable when augmenting traditional physiological baselines with Transformer-based embeddings. Furthermore, we emphasize algorithmic transparency through post-hoc explainability techniques to ensure clinical interpretability.

## Technical Context & Implementation Framework
The architecture of this predictive system relies on a synthesis of advanced natural language processing and robust gradient boosting algorithms. To tackle the inherent challenges of high-dimensional clinical text alongside strict computational limitations, we utilized a fine-tuned TinyClinicalBERT (`nlpie/tiny-clinicalbert`) model as the core NLP engine. This domain-specific architecture was explicitly selected over standard BERT computational equivalents to accommodate strict GPU memory constraints while preserving specialized medical vocabulary representations.

The knowledge extraction pipeline employs a Sequential/Late Fusion strategy. Initially, the fine-tuned language model processes clinical progress notes to generate a composite "Risk Score." This risk score serves as a dense feature representation of Latent Clinical Sentiment, mathematically synthesizing underlying narrative cues of clinical sentiment and physician uncertainty into a singular predictive signal. This dense feature is subsequently appended to the structured physiological data (vitals and lab results) prior to final classification.

The terminal classification function is executed via an XGBoost algorithm. To rigorously address the pathogenic class imbalance, we implemented a highly penalized Stochastic Gradient Descent framework by configuring the `scale_pos_weight` parameter to approximately 20. This optimization specifically targets the nuanced geometry of minority class (mortality) distributions, strictly optimizing Class Imbalance Bio-metrics such as sensitivity over deceptive generalized accuracy metrics.

## Technology Stack and Justification

*   **Core NLP Engine (Transformer Embeddings):** Hugging Face `transformers`, PyTorch, `nlpie/tiny-clinicalbert`.
    *   *Rationale:* TinyClinicalBERT was explicitly selected over standard BERT computational equivalents (e.g., `bert-base-uncased`) to operate within strict GPU memory constraints while fully preserving the specialized medical vocabulary and contextual representations necessary for analyzing clinical progress notes.
*   **Classification Framework:** XGBoost (`xgboost`), Scikit-learn.
    *   *Rationale:* XGBoost was chosen for its robust handling of structured tabular physiological data and its capability to enforce severe penalization on majority class classification errors. The `scale_pos_weight` parameter (set to ~20) was essential for aggressively optimizing Class Imbalance Bio-metrics in the highly skewed (1:20) MIMIC-III mortality cohort, actively prioritizing Recall over simple Accuracy.
*   **Explainability and Validation:** SHAP (`shap`).
    *   *Rationale:* Integrated to bridge the gap between complex algorithmic outputs (the fused NLP Risk Score) and actionable clinical decision-making. SHAP is essential for generating both global cohort-level insights (via Beeswarm plots) and highly granular, patient-specific mechanistic breakdowns (via Waterfall plots).
*   **Data Processing Pipeline:** Python, Pandas, NumPy, Jupyter Notebooks.
    *   *Rationale:* The standard, robust computational ecosystem for biomedical data science. It enables rapid interactive experimentation, seamless multimodal data structuring (Sequential/Late Fusion), and unified execution across the Baseline, Fine-Tuning, and Enhanced pipeline stages defined in the accompanying `.ipynb` files.

## Methodology and Results
The empirical evaluation of this framework is documented across three primary analytical stages, each corresponding to a specific implementation phase within the accompanying systematic notebooks (03 - Baseline, 04 - Fine-tune, and 05 - Enhanced). Standard evaluation metrics such as simple accuracy were fundamentally rejected due to the extreme class imbalance; consequently, formal assessment is strictly benchmarked using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) alongside Recall metrics.

The Baseline experimental phase (Notebook 03) relied exclusively on structured physiological vitals. While this methodology achieved a statistically moderate AUC of 0.728, it suffered from a critical failure in sensitivity, yielding a Recall of merely 0.09. This vulnerability highlights the inadequacy of purely numerical data in detecting subtle deteriorations in patient stability.

The Innovation phase, characterized by the Fine-tuning (Notebook 04) and Enhanced (Notebook 05) stages, seamlessly integrated the extracted NLP Risk Score. The introduction of this Latent Clinical Sentiment variable raised the overall predictive AUC to 0.771. More importantly, this multimodal fusion nearly quadrupled the Recall to 0.34. This radical improvement empirically proves that clinical notes contain crucial, "hidden" prognostic indicators of high-risk physiological decline that isolated laboratory values systematically miss.

### Summary Metrics
*   **Baseline Model (Structured Data Only):** AUC = 0.728, Recall = 0.09
*   **Enhanced Model (Multimodal Fusion):** AUC = 0.771, Recall = 0.34

## Explainability and Clinical Validation
To bridge the gap between complex algorithmic outputs and actionable clinical decision-making, we integrated SHAP (SHapley Additive exPlanations) as the primary mechanism for interpretability. Global feature behavior is visualized utilizing SHAP Beeswarm plots, which map the holistic impact of the Latent Clinical Sentiment and structured clinical variables across the entire patient cohort. Conversely, localized clinical validation is achieved through independent SHAP Waterfall plots, providing highly granular, patient-specific Mechanistic breakdowns of the predicted risk trajectory.

## Codebase and Repository Architecture
The project is structurally organized to ensure reproducibility across data generation, NLP fine-tuning, and algorithmic evaluation.

```text
IRP/
├── data/                    # Local raw and temporary data storage
├── dataset/                 # Persisted structured clinical data and derived text CSV files 
├── core/                    # Core library for utility scripts and custom helper functions
├── results/                 # Output directory for serialized models, metrics, and SHAP plots 
├── notebooks/               # Primary analytical execution pipeline (Jupyter)
│   ├── 03_Baseline.ipynb    # XGBoost physiological structural model baseline deployment
│   ├── 04_Fine_tune.ipynb   # TinyClinicalBERT NLP task-specific optimization pipeline
│   └── 05_Enhanced.ipynb    # Multimodal Sequential/Late Fusion and Explainability (SHAP)
├── .gitignore               # Ignored system files and heavy raw MIMIC-III data
└── README.md                # Project documentation and Technical Context
```

### Module Descriptions
*   **`notebooks/`**: The intellectual core of the repository. All statistical claims (AUC-ROC, Recall boosts) originate from the execution sequence defined within these interactive files.
*   **`results/`**: Designed to be the landing zone for the interpretability framework. High-resolution SHAP visual outputs (Beeswarm, Waterfall) and optimal model weights are written here during the `05_Enhanced` phase.
*   **`core/`**: Houses abstracted Python logic intended to prevent bloated code within the primary notebooks, ensuring data loading and metric formatting remain standardized across the baseline and enhanced pipelines.

## Limitations
*   The reliance on static historical cohort data emphasizes the necessity for temporal external validation.
*   The current Recall threshold (0.34), while demonstrating a quadrupled performance over baseline, indicates the persistent difficulty of capturing pathological anomalies within inherently skewed datasets.

## References
*   Acosta, J.N., Falcone, G.J. and Rajpurkar, P. (2022) 'Multimodal deep learning for healthcare', *Nature Reviews Bioengineering*, 1(1), pp. 1-14. Available at: https://doi.org/10.1038/s44222-022-00004-1.
*   Lundberg, S.M. et al. (2020) 'From local explanations to global understanding with explainable AI for trees', *Nature Machine Intelligence*, 2(1), pp. 56-67.
*   Soenksen, L.R. et al. (2022) 'Integrated multimodal artificial intelligence framework for healthcare applications', *npj Digital Medicine*, 5(1), p. 149. Available at: https://doi.org/10.1038/s41746-022-00689-4.
=======
# IRP

>>>>>>> b5ffbec (Initial commit with full IRP structure and LFS tracking)
=======
# IRP

>>>>>>> 532f0054829a2129c9ff39725d3278ed321e6dff
