"""TF-IDF từ/ngữ ký tự + Logistic Regression; học riêng từng phần, có mô hình cuối."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits
from .common_add import ASPECTS,LABELS,prediction_records,report_predictions
from .data_add import dataset,stamp,assert_identity
from src.training.prepare_foody_absa_add import save_json,sha


def vectorizer(cfg):
    return FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=cfg['word_features'],min_df=1)),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,max_features=cfg['char_features'],min_df=1))])

def aligned(model,x):
    raw=model.predict_proba(x);p=np.zeros((x.shape[0],len(LABELS)))
    for j,l in enumerate(model.classes_):p[:,LABELS.index(l)]=raw[:,j]
    return p

def fit(df,cfg):
    v=vectorizer(cfg);x=v.fit_transform(df.review_text_nlp);models={}
    for a in ASPECTS:
        m=DummyClassifier(strategy='most_frequent') if df[a].nunique()<2 else LogisticRegression(C=cfg['C'],max_iter=cfg['max_iter'],class_weight='balanced',random_state=42)
        models[a]=m.fit(x,df[a])
    return {'vectorizer':v,'classifiers':models,'labels':LABELS}

def run(root,cfg,final=False):
    df=dataset(root,cfg);out=root/cfg['report_dir']/'tfidf_add';out.mkdir(parents=True,exist_ok=True)
    assert_identity(out/'run_identity_add.json',stamp(root,cfg))
    if final:
        if not (out/'metrics_summary_add.json').exists():raise ValueError('Chạy tfidf CV trước khi refit.')
        md=root/'models/tfidf_foody_add';md.mkdir(parents=True,exist_ok=True)
        if (md/'training_complete_add.json').exists():
            completion=json.loads((md/'training_complete_add.json').read_text(encoding='utf-8'))
            if not (md/'model_add.joblib').exists() or sha(md/'model_add.joblib')!=completion['model_sha256']:raise ValueError('Mô hình TF-IDF cuối thiếu hoặc bị đổi.')
            return
        with threadpool_limits(limits=cfg['cpu_threads']): bundle=fit(df,cfg['tfidf'])
        joblib.dump(bundle,md/'model_add.joblib',compress=3)
        df[['review_id','text_group_id']].to_csv(md/'training_review_ids_add.csv',index=False,encoding='utf-8-sig')
        save_json(md/'training_complete_add.json',{'train_reviews':len(df),'evaluation_performed':False,'identity':stamp(root,cfg),'model_sha256':sha(md/'model_add.joblib')})
        return
    cv=df[df.fold.gt(0)];records=[];dummy=[]
    for f in range(1,6):
        fold=out/f'fold_{f}_add';fold.mkdir(exist_ok=True);pred=fold/'predictions_add.csv';done=fold/'complete_add.json'
        if done.exists():
            records.extend(pd.read_csv(pred,keep_default_na=False).to_dict('records'))
            dummy.extend(pd.read_csv(fold/'majority_add.csv',keep_default_na=False).to_dict('records'));continue
        tr,te=cv[cv.fold.ne(f)],cv[cv.fold.eq(f)]
        with threadpool_limits(limits=cfg['cpu_threads']):bundle=fit(tr,cfg['tfidf'])
        x=bundle['vectorizer'].transform(te.review_text_nlp);r=[];d=[]
        for a in ASPECTS:
            r+=prediction_records(te,a,aligned(bundle['classifiers'][a],x),'tfidf_add',f)
            majority=tr[a].value_counts().idxmax();pr=np.zeros((len(te),5));pr[:,LABELS.index(majority)]=1
            d+=prediction_records(te,a,pr,'majority_add',f)
        pd.DataFrame(r).to_csv(pred,index=False,encoding='utf-8-sig')
        pd.DataFrame(d).to_csv(fold/'majority_add.csv',index=False,encoding='utf-8-sig')
        if cfg['save_fold_models']:
            md=root/f'models/tfidf_foody_add/fold_{f}_add';md.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,md/'model_add.joblib',compress=3)
        save_json(done,{'fold':f,'train':len(tr),'test':len(te),'test_used_for_selection':False})
        records+=r;dummy+=d;print(f'TF-IDF phần {f}/5 đã xong.',flush=True)
    report_predictions(pd.DataFrame(records),out,cv);report_predictions(pd.DataFrame(dummy),out/'majority_add',cv)
