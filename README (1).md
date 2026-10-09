# Financial Fraud Detection with Machine Learning

A reproducible academic portfolio project: real anonymized credit-card transactions, chronological evaluation, class imbalance handling, validation-based threshold selection, and a Streamlit review dashboard.

## Research question
Can a supervised classifier detect rare fraud while keeping false alerts manageable on later transactions?

## Dataset and provenance
ULB / Worldline credit-card fraud dataset via [OpenML 1597](https://www.openml.org/d/1597), also available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
The source contains 284,807 transactions and 492 frauds over two days. Features: Time, Amount, anonymized PCA variables V1–V28; target Class (1 fraud). It is an older benchmark, not 2026 transaction data. It has no customer IDs or interpretable merchant attributes. Check source terms before redistributing; raw data is excluded from GitHub.

## Run locally (Python 3.11 or 3.12)
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
python train.py
python -m streamlit run app.py
```
If OpenML is unreachable, download creditcard.csv from Kaggle and place it alongside train.py. No synthetic fallback is silently substituted.

## Methodology
1. Validate numeric values, schema and binary labels; remove identical rows and record their count. Identical anonymized observations may be distinct transactions, so duplicate removal is a documented assumption.
2. Sort by Time; time quantile boundaries create approximately 60% training, 20% validation, 20% test. Equal timestamps stay together.
3. Fit training-only scaling for balanced logistic regression; compare against balanced random forest and a prior dummy baseline. No resampling changes validation/test prevalence.
4. Select the supervised model using validation Average Precision (AP); select its threshold using validation F2, favoring recall.
5. Evaluate the selected, unchanged training model once on the later test partition. Report AP, ROC AUC, precision, recall, F1, confusion matrix, alert rate and false alerts per 1,000 transactions.
6. Export model, metrics, test predictions and plots; show review queue and CSV scoring in dashboard.

AP is the step-weighted precision-recall summary, not trapezoidal PR area. A model always predicting legitimate can achieve about 99.83% accuracy on the source data, so accuracy alone is misleading. F2 is an educational policy choice, not a bank's measured business cost.

## Outputs
- artifacts/metrics.json: split sizes, fraud counts, validation comparison and held-out test metrics
- artifacts/model.joblib: selected pipeline and threshold (load only your own trusted artifacts)
- artifacts/test_predictions.csv: held-out labels, risk scores and alerts
- artifacts/*.png: precision-recall, ROC, confusion matrix and class distribution
- sql/analysis.sql: optional SQLite review queries
- notebooks/walkthrough.ipynb: EDA and reproducible workflow

## Results from the real dataset run

The model was selected by validation Average Precision, and its decision threshold was selected using validation F2. The later chronological test partition was evaluated once.

| Metric | Held-out test result |
|---|---:|
| Selected model | Random Forest |
| Average Precision | 0.8151 |
| ROC AUC | 0.9353 |
| Precision | 0.9167 |
| Recall | 0.7432 |
| F1 | 0.8209 |
| Decision threshold | 0.2731 |
| False alerts per 1,000 transactions | 0.0881 |

Confusion matrix on 56,746 test transactions: TN=56,667, FP=5, FN=19, TP=55. The model detected 55 of 74 fraud cases and missed 19; it flagged 5 legitimate transactions.

Validation AP comparison: dummy 0.0010, logistic regression 0.7727, random forest 0.7770. Random forest was selected because it had the highest validation AP. Full run details are in `artifacts/metrics.json`.

These are results from one historical benchmark split, not a production guarantee. The dataset contains transactions from two days and uses anonymized PCA features.

## Limits and extensions
Only two days of historical data; no account IDs for account-level separation, no label-delay simulation, and anonymized PCA transformation predates this experiment. Time holdout therefore does not guarantee a fully leakage-free real deployment study. No probability calibration or independent external validation. Inspect false negatives, compare time exclusion, repeat over rolling windows, add calibration with separate validation, and evaluate a documented review-capacity threshold. Do not repeatedly tune against this test set.

## Academic ownership
Prepared with AI assistance. Read, run, modify and explain the code before presenting it as your work. Record your own experiments in an experiment log; do not claim production deployment, banking experience, or fabricated metrics.

## References
- https://www.openml.org/d/1597
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- https://scikit-learn.org/stable/modules/model_evaluation.html
