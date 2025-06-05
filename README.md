Deep Learning for Credit Default Prediction: A Multi-Modal Approach with Tabular and Text Data
Project Overview
This project develops and evaluates a hybrid deep learning framework for predicting credit default in unsecured personal lending, with a specific focus on the rapidly digitizing financial services sector in Vietnam. The primary goal is to enhance prediction accuracy, model interpretability, and real-world applicability in credit scoring.

The proposed solution integrates multi-modal data sources—structured tabular data and unstructured textual data—leveraging state-of-the-art deep learning architectures: TabTransformer for tabular features, FinBERT for financial text narratives, and Concept Bottleneck Models (CBMs) to ensure human-aligned interpretability. This approach aims to address critical challenges in credit risk modeling, including handling heterogeneous data, severe class imbalance, the demand for explainable AI (XAI), and the need for real-time decision-making capabilities in digital lending platforms.

Table of Contents
Motivation
Proposed Solution
Model Architecture
Tabular Data Processing
Textual Data Processing (NLP)
Concept Bottleneck Models (CBMs)
Multi-Modal Fusion
Dataset
Key Features and Innovations
Experimental Setup
Training Process
Evaluation Metrics
Results
Proposed Model Performance
Interpretability Analysis (SHAP)
Hyperparameter Tuning
Conclusion
Future Work
Setup and Usage
License
Citation
Motivation
Traditional credit scoring methodologies often rely on structured financial records and perform inadequately for "credit invisibles"—individuals lacking verifiable income, formal employment, or collateral, especially in emerging markets like Vietnam. This exclusion hinders financial inclusion and broader economic goals. Fintech lenders are leveraging AI/ML with alternative data, including textual inputs, to enhance risk assessment. However, the "black box" nature of many deep learning models poses risks regarding accountability, bias, and regulatory compliance. There's a pressing need for models that are not only accurate but also transparent, fair, and interpretable, particularly in high-stakes financial domains. This project addresses these multidimensional challenges.





Proposed Solution
Model Architecture
The project proposes a hybrid deep learning architecture that synergistically combines TabTransformer for structured tabular data, FinBERT for unstructured textual inputs, and a Concept Bottleneck Layer (CBM) for interpretable reasoning. These components converge in a multi-modal fusion layer, feeding into a final classification head to produce a default probability.




Figure 1: The hybrid model architecture 
Raw Tabular Features -> TabEncoder -> TabTransformer Encoder 
Loan Purpose Text -> FinBERT Text Encoder -> Text Projection Layer 
CBM Features -> CBM Projection 
All three paths merge into a "Final Tabular Embedding" -> Final Classifier -> Output: Default Probability.
Tabular Data Processing
Tabular Encoder - TabTransformer: Structured borrower data (e.g., income, debt, employment status) is processed using TabTransformer.

Categorical features are embedded into token vectors.
Numerical features are normalized and passed directly.
A stack of transformer encoder blocks learns feature-to-feature dependencies.
Figure 2: TabTransformer diagram 
Numerical Feature Processing:

Continuous variables (e.g., monthly_income, total_outstanding_debt) are normalized using StandardScaler. Outliers are retained.

Categorical Feature Encoding:

Binary encoding for dichotomous variables (e.g., has_bank_account_linked).
Ordinal encoding for ordered variables.
Learnable embeddings for higher-cardinality fields within TabTransformer.
Missing values handled by a "missing" category label.
Domain-Informed Feature Construction:

Derived features like debt_to_income_ratio and expense_to_income were engineered.
loan_count_score and risk_score_raw are used as behavioral metrics/signals.
Textual Data Processing (NLP)
Text Encoder - FinBERT: The loan_purpose_text field is embedded using FinBERT, a transformer model pretrained on financial language corpora. 

FinBERT captures contextual semantics and financial sentiment.
It distinguishes nuanced intentions and outputs a 768-dimensional vector per input.
Semantic Preprocessing with TF-IDF:
Initially, Term Frequency-Inverse Document Frequency (TF-IDF) was used to extract relevant semantic keywords from loan justifications. This helped in preliminary data exploration and validation.

Concept Bottleneck Models (CBMs)
Concept Bottleneck Layer: To enhance explainability, the model includes a CBM layer that predicts intermediate, human-understandable concepts before final prediction.



Engineered CBM Features:
cbm_feature_1: Composite Behavioral Risk Indicator, derived from total_late_payments, num_late_payments_in_app, loan_count_score, number_of_current_loans, and risk_score_raw. This reflects cumulative behavioral risk.


cbm_feature_2: Financial Stability Proxy, based on expense_to_income ratio, employment status (from employment_status, job_type, is_unemployed), and potentially text-derived volatility cues. This approximates financial buffer and employment security.


These concepts provide semantic transparency, modular accountability, and support bias detection.


Multi-Modal Fusion
Latent representations from TabTransformer, FinBERT, and CBM modules are concatenated and passed through a fully connected neural network with a sigmoid activation for final probability prediction. The prediction is P(Default=1∣X_tab,X_text,X_cbm).



Dataset
Source and Collection: Compiled from three Vietnamese commercial banks and two leading fintech companies operating in unsecured digital lending. Data spans January 2021 to December 2024. Loans are collateral-free, reflecting real-time approval for "thin-file" borrowers.


Size and Labeling: 400,000 anonymized loan application records. 

Target variable: default = 1 if borrower defaulted, default = 0 if repaid.

Approximately 21% of borrowers are labeled as defaulters, reflecting natural class imbalance. No artificial resampling was applied to preserve this distribution.



Data Structure Overview: 26 variables (24 original, 2 engineered CBM features) across demographic, financial, behavioral, and textual categories. 
Table 1: Demographic Variables (age, gender, marital_status, residential_area, employment_status, job_type) 
Table 2: Financial Variables (monthly_income, estimated_monthly_expense, total_outstanding_debt, bank_avg_balance, has_bank_account_linked, has_e_wallet) 
Table 3: Behavioral & Credit History Features (number_of_current_loans, total_late_payments, num_loans_from_app, num_late_payments_in_app, loan_count_score, risk_score_raw, wallet_usage_frequency, debt_to_income_ratio, expense_to_income, is_unemployed) 
Table 4: Textual Input (loan_purpose_text) 
Table 5: Target Variable (default) 
Table 6: Concept Bottleneck Features (Engineered) (cbm_feature_1, cbm_feature_2) 
Anonymization: All data was fully anonymized according to Vietnamese regulations and international standards; no PII was retained.
Key Features and Innovations
Multi-Modal Learning: Integrates structured tabular data and unstructured text to capture a richer representation of borrower profiles.

Advanced Deep Learning Models: Utilizes TabTransformer for nuanced understanding of tabular feature interactions and FinBERT for domain-specific financial text understanding.

Explainable AI (XAI): Incorporates Concept Bottleneck Models (CBMs) for inherent interpretability by mapping predictions through human-understandable concepts. Complemented by SHAP for feature attribution analysis.



Class Imbalance Handling: Addresses severe class imbalance through a weighted loss function rather than data resampling, preserving the natural data distribution.

Real-Time Applicability: Designed with considerations for low-latency inference suitable for automated, real-time loan approval systems.


Contextual Relevance: Tailored to the specific challenges and data landscape of the Vietnamese unsecured lending market, focusing on financial inclusion for "thin-file" borrowers.

Experimental Setup
Data Splitting: 80% training set (320,000 samples) and 20% testing set (80,000 samples), using stratified splitting to maintain class distribution. No synthetic oversampling or undersampling.



Feature Modalities: Structured tabular features, textual features (loan_purpose_text), and engineered concept bottleneck features.
Model Components: TabTransformer, FinBERT, and CBM integrated into a PyTorch model.

Computing Environment: Kaggle Notebooks with a Tesla P100 GPU (16GB VRAM). 

Python version used in the project (the document states 3.13.2, which might be a typo; users should specify the actual version).
Key libraries: PyTorch, Transformers, scikit-learn, pandas, seaborn, matplotlib.

Hyperparameters (Initial):
TabTransformer: 2 encoder layers, 4 attention heads, d_model=128, dropout=0.1.

FinBERT pretrained weights: yiyanghkust/finbert-tone.

CBM projection: 2 input features -> Linear -> ReLU -> Dropout(0.1).

Fusion layer described.

Training Process
Optimizer: AdamW optimizer. 
Learning rate:
eta=2
times10 
−4
 .


Weight decay:
lambda=1
times10 
−4
 .


Learning Rate Scheduler: ReduceLROnPlateau with patience=4, factor=0.5, based on validation loss plateauing.
Loss Function: Weighted Binary Cross-Entropy with Logits (nn.BCEWithLogitsLoss(pos_weight=3.6) in PyTorch) to handle class imbalance, where pos_weight (3.6) penalizes false negatives more heavily.

Regularization:
Dropout after each dense layer (P=0.1 or 0.3 as mentioned, e.g., 0.3 in source , 0.1 in source ). (Clarify from code).



Early stopping triggered after 8 consecutive epochs without improvement in test loss.

Epochs: Trained for up to 150 epochs.


Batch Size: 128  (later tuned, see Hyperparameter Tuning section).
Training procedure: Supervised deep learning, mixed precision (float32) on GPU.

Evaluation Metrics
A comprehensive set of metrics was used due to class imbalance and the importance of interpretability.

Classification Metrics: 
Accuracy 
Precision 
Recall (Sensitivity) 
F1-score 
AUC-ROC (Area Under the Receiver Operating Characteristic curve) 
PR AUC (Precision-Recall AUC), more informative for skewed datasets 
Confusion Matrix 
Calibration Metrics: To evaluate the reliability of probability estimates. 
Brier Score 
Expected Calibration Error (ECE) 
Explainability and Concept-Level Evaluation: 
SHAP (SHapley Additive exPlanations): For feature-level attribution, global importance, and local explanations.
Concept Agreement (CBM Validation): Internal CBM values compared against domain-driven reasoning paths, with expert audit of sample instances.
Results
Proposed Model Performance (Tabtransformer + FinBERT + CBMs)
The performance of the hybrid model was extensively evaluated.

Classification Performance (Table 8): 
Overall accuracy 98%.


Weighted average Precision 0.98, Recall 0.97, F1-score 0.98.

Balanced F1-scores: 0.98 for non-default (class 0), 0.99 for default class (class 1).

Confusion Matrix Analysis (Figure 6): 
The confusion matrix for the hybrid model (when optimized for accuracy) showed that while overall metrics were high, it resulted in a high number of false negatives for the default class (16,749 out of 16,800 actual defaults were misclassified as non-defaults). This suggests a strong skew towards optimizing global accuracy rather than minority class sensitivity with the current final-layer threshold calibration. Recalibration strategies could be explored for better minority class capture.



Training vs. Testing Loss (Figure 7): 
Both training and testing loss exhibited a consistent downward trend, especially during the first 80 epochs.
From approximately epoch 100 onward, both curves flattened out and stabilized, suggesting convergence with minimal overfitting, indicating good generalization.

ROC and Precision-Recall Curves (Figure 8): 
ROC AUC = 0.55: The ROC curve lies only slightly above the random classifier baseline. This indicates the model barely outperforms random guessing in distinguishing between classes based on this metric, which can be less informative in severely imbalanced settings.

PR AUC = 0.57: The Precision-Recall curve shows a steep drop in precision as recall increases, reflecting modest performance in identifying true default cases. Since defaults are the minority class, PR AUC is considered more appropriate, and its score indicates insufficient recall-oriented learning with the current setup. (Note: The high F1 for the default class reported in Table 8  appears to contrast with the low PR AUC  and the high number of False Negatives shown in the confusion matrix analysis for Figure 6. This suggests that the F1 score in Table 8 might be based on a different operating point or threshold than what is reflected in the PR curve or the specific confusion matrix presented.)




Interpretability Analysis (SHAP)
SHAP Summary Plot (Figure 9):  Illustrates the impact of each input feature on default prediction. 

Top Influential Features:
risk_score_raw: Most influential; high values strongly predict default.
is_unemployed: Strong positive effect on predicted default.
total_outstanding_debt and wallet_usage_frequency: Meaningful impact; high debt increases risk, low wallet activity may reflect weak engagement.
total_late_payments: Higher values lead to greater risk.
CBM Validation: cbm_feature_1 (composite risk indicator) appears in top features and positively correlates with risk, validating CBM's role in transparency.
Fairness Indication: Features like gender and residential_area have relatively low SHAP impact, suggesting the model does not heavily rely on these potentially sensitive attributes.
Overall, model behavior aligns with domain expectations, and the interpretability from SHAP and CBMs supports transparent and auditable deployment.
Hyperparameter Tuning
Systematic tuning was performed to balance learning capacity, generalization, and efficiency.

Learning Rate: Selected 2
times10 
−4
  from range [1e-5, 2e-4] for fast convergence and stability.
Batch Size: 128 and 256 tested; the final model used 128 for training (as per general setup) but 256 was noted as providing a good trade-off. (Clarify from actual final run for consistency).
Dropout Rate: Explored 0.05-0.2; final model uses 0.1.
Number of Transformer Layers (Tabular): Two TransformerEncoder layers used for enhanced representational power.
d_model (Hidden Dimension Size): Set to 128 (compared 64, 128, 256) for balance.
Optimizer and Scheduler: AdamW with ReduceLROnPlateau.
Early Stopping: Patience of 8 epochs on validation loss.
Class Weighting: Balanced class weights applied via pos_weight in BCEWithLogitsLoss.
Tuning guided by validation loss, AUC, and PR performance on hold-out set.
Conclusion
The proposed hybrid deep learning architecture (TabTransformer + FinBERT + CBMs) achieved high overall predictive performance (accuracy 
approx98, macro-average F1-score 
approx0.97) for credit default prediction in Vietnam's unsecured lending sector. It successfully integrated multi-modal data and delivered human-aligned explainability via CBMs and semantic understanding via FinBERT, with SHAP analysis confirming domain-aligned feature importance. While some metrics (PR AUC, specific confusion matrix FNs) highlighted challenges with minority class recall under certain optimization goals, the model's framework addresses key industry needs for transparent, interpretable, and high-performing AI in finance. This research contributes a replicable blueprint for ethically grounded and regulatory-aligned credit scoring systems, particularly for emerging markets.




Future Work
Fairness-Aware Modeling: Systematically evaluate and mitigate group-wise disparities using fairness metrics and techniques like adversarial debiasing or fairness-constrained optimization.
Explainability Beyond CBMs: Integrate complementary tools like counterfactual explanations or causality-based frameworks.
Dynamic Credit Scoring & Personalized Pricing: Evolve into a real-time, multi-level risk scoring framework for dynamic interest rate personalization.
Multilingual and Cross-Market Adaptation: Adapt the architecture to other Southeast Asian markets by retraining language modules and CBMs for regional contexts.
Integration with Credit Lifecycle Systems: Extend to support fraud detection, early delinquency prediction, and collections strategy optimization within LMS/CRM platforms.
