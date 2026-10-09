# Execution status

- Python source compilation passed in the build environment.
- Training, validation model selection, threshold selection, metric export and plot generation passed on a 1,500-row synthetic execution fixture. Synthetic outputs were removed.
- The user subsequently ran the full project on the real OpenML 1597 / ULB Worldline dataset. Dashboard screenshot: 284,807 raw rows, 1,081 duplicates removed; chronological partitions had 170,235 train, 56,745 validation and 56,746 test rows. Test confusion matrix: TN 56,667, FP 5, FN 19, TP 55. Displayed test AP 0.8151, precision 0.9167, recall 0.7432 and 0.0881 false alerts per 1,000 transactions.
- The uploaded ZIP did not include generated artifacts/metrics.json, fitted model or plots. Run `python train.py` to recreate them.
- Interactive Streamlit dashboard was verified by the user on their computer at localhost:8501.
