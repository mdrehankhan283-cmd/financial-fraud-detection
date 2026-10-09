"""Chronological holdout; validation-only model and threshold selection."""
from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, precision_recall_curve, PrecisionRecallDisplay,
    RocCurveDisplay, ConfusionMatrixDisplay)
ROOT = Path(__file__).parent
FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
def validate(df):
    required = FEATURES + ['Class']
    if set(required) - set(df): raise ValueError('Missing required columns')
    df = df[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df.to_numpy()).all(): raise ValueError('Missing or nonfinite values')
    if not set(df.Class.unique()) <= {0, 1}: raise ValueError('Class must be 0 or 1')
    if (df.Amount < 0).any(): raise ValueError('Amount cannot be negative')
    return df

def main(path, trees):
    out = ROOT / 'artifacts'; out.mkdir(exist_ok=True)
    raw = validate(pd.read_csv(path))
    # Remove identical rows before splitting so duplicates cannot cross splits.
    df = raw.drop_duplicates().sort_values('Time', kind='stable').reset_index(drop=True)
    # Keep equal timestamps together; use time quantiles rather than random split.
    a, b = df.Time.quantile([.6, .8]).tolist()
    train = df[df.Time < a]; val = df[(df.Time >= a) & (df.Time < b)]; test = df[df.Time >= b]
    for part in (train, val, test):
        if part.Class.nunique() != 2: raise ValueError('Each split needs both classes')
    models = {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic': make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1500, random_state=42)),
        'random_forest': RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    }
    results = []; chosen = None; best = -1
    for name, model in models.items():
        model.fit(train[FEATURES], train.Class)
        scores = model.predict_proba(val[FEATURES])[:, 1]
        ap = average_precision_score(val.Class, scores)
        results.append({'model': name, 'validation_AP': float(ap)})
        if name != 'dummy' and ap > best: chosen = (name, model, scores); best = ap
    name, model, scores = chosen
    precision, recall, thresholds = precision_recall_curve(val.Class, scores)
    # F2 weights recall more than precision; no test labels influence this choice.
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[np.argmax(f2)])
    prob = model.predict_proba(test[FEATURES])[:, 1]; pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test.Class, pred, labels=[0, 1]).ravel()
    metrics = {'model':name, 'threshold':threshold, 'average_precision':float(average_precision_score(test.Class, prob)),
        'roc_auc':float(roc_auc_score(test.Class, prob)), 'precision':float(precision_score(test.Class,pred,zero_division=0)),
        'recall':float(recall_score(test.Class,pred,zero_division=0)), 'f1':float(f1_score(test.Class,pred,zero_division=0)),
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp), 'false_alerts_per_1000':float(fp/len(test)*1000),
        'alert_rate':float(pred.mean()), 'fraud_prevalence':float(test.Class.mean())}
    split_info = {label: {'rows':len(part),'frauds':int(part.Class.sum()),'time_min':float(part.Time.min()),'time_max':float(part.Time.max())}
        for label,part in [('train',train),('validation',val),('test',test)]}
    report = {'source':'OpenML 1597 / ULB Worldline', 'raw_rows':len(raw),'duplicates_removed':len(raw)-len(df),
        'split':split_info,'validation_comparison':results,'test_metrics':metrics,
        'policy':'Model chosen by validation AP; threshold by validation F2; model remains trained on training partition only.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    joblib.dump({'model':model,'threshold':threshold,'features':FEATURES},out/'model.joblib')
    export = test.copy(); export['risk_score']=prob; export['alert']=pred; export.to_csv(out/'test_predictions.csv',index=False)
    for filename, display in [('precision_recall', PrecisionRecallDisplay.from_predictions(test.Class,prob)),
        ('roc',RocCurveDisplay.from_predictions(test.Class,prob)),
        ('confusion_matrix',ConfusionMatrixDisplay.from_predictions(test.Class,pred,labels=[0,1]))]:
        display.figure_.savefig(out/f'{filename}.png',bbox_inches='tight'); plt.close(display.figure_)
    fig,ax=plt.subplots(); df.Class.value_counts().sort_index().plot.bar(ax=ax,logy=True)
    ax.set(xlabel='Class (0 legitimate, 1 fraud)',ylabel='Transactions (log scale)',title='Class imbalance')
    fig.savefig(out/'class_balance.png',bbox_inches='tight'); plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data',default=str(ROOT/'creditcard.csv'))
    parser.add_argument('--trees',type=int,default=150); args=parser.parse_args(); main(args.data,args.trees)
