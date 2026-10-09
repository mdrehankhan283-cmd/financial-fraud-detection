# Step-by-step guide — Rehan ka admission portfolio

1. ZIP extract karo. Python 3.11/3.12 install karo; VS Code mein extracted folder kholo. Terminal mein README ke commands chalao. Windows par `python` na chale to `py` use karo. Activation blocked ho to `.venv\Scripts\python.exe -m pip install -r requirements.txt` aur isi interpreter se scripts chalao.
2. `python download_data.py` real dataset download karega. Download fail ho to README ka Kaggle link use karke creditcard.csv folder mein rakho.
3. Notebook kholo: `python -m pip install jupyter` phir `python -m jupyter notebook`. EDA cells run karo, class imbalance aur Amount distribution samjho.
4. `python train.py` run karo. Runtime laptop par depend karta hai; training ke dauran terminal khula rakho. Output artifacts folder mein aayega.
5. metrics.json mein actual AP, precision, recall, FP aur FN dekho. Fraud ko miss karna FN hai; legitimate transaction ko flag karna FP hai. High recall ka matlab zyada fraud pakadna; precision ka matlab alerts mein kitne sahi fraud hain.
6. `python -m streamlit run app.py` chalao. Terminal ka local URL browser mein kholo. Dashboard screenshots lo. CSV scoring ke liye benchmark ki original feature columns chahiye; ordinary account records ka format is model ke compatible nahi hai.
7. GitHub par public repository `financial-fraud-detection` banao. README.md root mein rakho. Source files, requirements, notebook, SQL aur GUIDE upload karo. creditcard.csv, .venv aur model.joblib upload mat karo. Real run ke metrics.json aur charts results ke saath add karo; raw prediction CSV public upload karne se pehle source terms check karo.
8. README mein apne actual results aur screenshots add karo. Repository link `https://github.com/YOUR_USERNAME/financial-fraud-detection` hoga. Is link ko CV ke project section mein use karo.
9. Apni experiment diary likho: baseline vs forest; 0.5 vs selected threshold; Time feature remove karne ka effect. Test dekh kar tune mat karo; further experiments ke liye separate holdout define karo.

## Interview mein samjhane ke points
- Problem: fraud rare hai, isliye accuracy misleading hai.
- Data: real anonymized transactions; purana two-day benchmark; current banking data nahi.
- Split: earlier training, middle validation, later test; future labels training mein use nahi kiye.
- Class weight: training algorithm fraud cases ko zyada importance deta hai.
- Scaling: logistic regression ke liye scaler sirf training data par fit hua.
- Threshold: validation F2 se select; test data se choose nahi kiya.
- Metrics: AP ranking quality; precision alerts ki quality; recall fraud coverage; confusion matrix errors.
- Limit: PCA V1–V28 ka real-world meaning unknown; model score calibrated probability nahi.

## CV wording — sirf real run aur understanding ke baad
Financial Fraud Detection | Python, scikit-learn, SQL, Streamlit
Built and evaluated a credit-card fraud classification pipeline with chronological holdouts, class-weighted learning and validation-selected alert thresholds; developed a dashboard for risk-score review and error analysis.
Actual measured metrics add kar sakte ho; 99% accuracy ya production deployment claim mat likho bina evidence.

Ye portfolio project hai; admission guarantee nahi. IMT interview mein apni decisions, actual results aur limitations confidently explain karna zaroori hai.
