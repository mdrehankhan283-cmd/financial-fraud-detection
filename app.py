from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
ROOT=Path(__file__).parent; OUT=ROOT/'artifacts'
st.set_page_config(page_title='Fraud Risk Analytics',layout='wide')
st.title('Financial Fraud Detection — Research Dashboard')
st.caption('Historical anonymized benchmark; risk scores are model outputs, not calibrated fraud probabilities.')
if not (OUT/'model.joblib').exists():
    st.info('Run python train.py first.'); st.stop()
bundle=joblib.load(OUT/'model.joblib')
report=json.loads((OUT/'metrics.json').read_text()); m=report['test_metrics']
cols=st.columns(4)
for col,key in zip(cols,['average_precision','precision','recall','false_alerts_per_1000']): col.metric(key,round(m[key],4))
st.write('Validation-selected decision threshold:',bundle['threshold'])
st.dataframe(pd.DataFrame(report['validation_comparison']),hide_index=True)
st.json(report['split'])
a,b=st.columns(2); a.image(str(OUT/'precision_recall.png')); b.image(str(OUT/'confusion_matrix.png'))
df=pd.read_csv(OUT/'test_predictions.csv')
st.subheader('Test-set review queue')
st.dataframe(df.sort_values('risk_score',ascending=False).head(100),hide_index=True)
st.subheader('Score a transaction CSV')
st.caption('Requires Time, Amount, V1–V28. PCA features must use the benchmark schema; ordinary raw banking records cannot be substituted.')
upload=st.file_uploader('CSV',type='csv')
if upload:
    try:
        incoming=pd.read_csv(upload); features=bundle['features']
        x=incoming[features].apply(pd.to_numeric,errors='raise')
        if not np.isfinite(x.to_numpy()).all() or (x.Amount<0).any(): raise ValueError('Invalid numeric values')
        incoming['risk_score']=bundle['model'].predict_proba(x)[:,1]
        incoming['alert']=(incoming.risk_score>=bundle['threshold']).astype(int)
        st.dataframe(incoming,hide_index=True)
        st.download_button('Download predictions',incoming.to_csv(index=False),'predictions.csv','text/csv')
    except (ValueError,KeyError) as error: st.error(f'Invalid input: {error}')
