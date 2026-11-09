"""Kiểm tra VLSP, chia nhóm, tạo token và vị trí bằng chứng dùng chung."""
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.training.prepare_foody_absa_add import sha, save_json
from src.training.group_folds_add import duplicate_components
from .common_add import ASPECTS, LABELS


def read_csv(path):return pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
def key(text):return ' '.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFC',text).casefold()))
def stamp(root,cfg):
    return {'data_sha256':sha(root/cfg['data_dir']/'absa_all_add.csv'),
            'config':cfg,'code_sha256':{str(p.relative_to(root)):sha(p) for p in sorted((root/'src').rglob('*_add.py'))}}

def assert_identity(path,identity):
    path=Path(path)
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=identity:
        raise ValueError(f'Dữ liệu/cấu hình/mã nguồn đổi so với {path}. Không nối tiếp lần chạy khác.')
    save_json(path,identity)


def audit(root,cfg):
    base=root/cfg['data_dir'];df=read_csv(base/'absa_all_add.csv');dest=base/'source_overlap_audit_add.json'
    source=root/cfg['vlsp_processed_dir'];paths={s:source/(s+'.csv') for s in ['train','dev','test']}
    for p in paths.values():
        if not p.is_file():raise FileNotFoundError(f'Thiếu VLSP đã xử lý {p}. Sửa vlsp_processed_dir; không bỏ qua kiểm tra trùng.')
    signature={'foody':sha(base/'absa_all_add.csv'),'vlsp':{s:sha(p) for s,p in paths.items()},'threshold':cfg['near_threshold']}
    if dest.exists():
        old=json.loads(dest.read_text(encoding='utf-8'))
        if old['signature']!=signature:raise ValueError('Nguồn VLSP hoặc Foody đã đổi; tạo lại đầu ra _add có chủ ý.')
        return old
    keys=df.review_text_nlp.map(key).tolist();lookup={}
    for i,k in enumerate(keys):lookup.setdefault(k,[]).append(i)
    matches=[]
    for split,p in paths.items():
        s=read_csv(p)
        textcol='review_text_raw' if 'review_text_raw' in s else 'Review'
        if textcol not in s:raise ValueError(f'{p} thiếu review_text_raw/Review')
        skeys=s[textcol].map(key).tolist()
        for i,k in enumerate(skeys):
            for j in lookup.get(k,[]): matches.append({'review_id':df.iloc[j].review_id,'split':split,'kind':'exact','source_row':i+2})
        if split in ['train','dev'] and skeys:
            v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),dtype=np.float32)
            m=v.fit_transform(skeys+keys);a,b=m[:len(skeys)],m[len(skeys):]
            for start in range(0,len(skeys),64):
                sim=(a[start:start+64]@b.T).tocoo()
                for x,y,score in zip(sim.row,sim.col,sim.data):
                    if score>=cfg['near_threshold'] and skeys[start+x]!=keys[y]:
                        matches.append({'review_id':df.iloc[y].review_id,'split':split,'kind':'near','similarity':float(score),'source_row':int(start+x)+2})
    result={'signature':signature,'matches':matches,'excluded_cv_review_ids':sorted({x['review_id'] for x in matches if x['split'] in ['train','dev']}),
            'policy':'Không chấm mẫu đã thấy trong VLSP train/dev; vẫn giữ khi huấn luyện cuối; không mặc định số mẫu trùng.'}
    save_json(dest,result);print(f"Trùng VLSP train/dev: {len(result['excluded_cv_review_ids'])} nhận xét.",flush=True);return result


def folds(root,cfg):
    base=root/cfg['data_dir'];source=base/'absa_all_add.csv';target=base/'fold_assignments_add.csv'
    identity={'source':sha(source),'seed':cfg['seed'],'audit':sha(base/'source_overlap_audit_add.json')}
    assert_identity(base/'split_identity_add.json',identity)
    if target.exists():return
    df=read_csv(source);groups,near=duplicate_components(df);df['split_group_id']=groups
    excluded=set(json.loads((base/'source_overlap_audit_add.json').read_text(encoding='utf-8'))['excluded_cv_review_ids'])
    excluded_groups=set(df.loc[df.review_id.isin(excluded),'split_group_id'])
    cv=df[~df.split_group_id.isin(excluded_groups)].copy()
    if cv.split_group_id.nunique()<5:raise ValueError('Cần ít nhất 5 nhóm sạch để kiểm tra chéo.')
    cols=[(a,l) for a in ASPECTS for l in LABELS];vectors={};indices={}
    for gid,g in cv.groupby('split_group_id'):
        indices[gid]=g.index.tolist();vectors[gid]=np.array([g[a].eq(l).sum() for a,l in cols])
    totals=sum(vectors.values());weights=1/np.sqrt(np.maximum(totals,1));weights[::5]*=.25
    ids=list(vectors);rng=np.random.default_rng(cfg['seed']);rng.shuffle(ids)
    ids.sort(key=lambda g:float((vectors[g]*weights).sum()),reverse=True)
    counts=np.zeros((5,len(cols)));sizes=np.zeros(5);assignment={};cap=int(np.ceil(len(cv)/5))
    for g in ids:
        n=len(indices[g]);candidate=[f for f in range(5) if sizes[f]+n<=cap] or list(range(5))
        f=max(candidate,key=lambda f:(float((vectors[g]*weights*(totals/5-counts[f])).sum())-.15*sizes[f],-sizes[f],-f))
        assignment[g]=f+1;counts[f]+=vectors[g];sizes[f]+=n
    df['fold']=df.split_group_id.map(assignment).fillna(0).astype(int)
    assert df.groupby('split_group_id').fold.nunique().max()==1
    df[['review_id','text_group_id','split_group_id','fold']].to_csv(target,index=False,encoding='utf-8-sig')
    save_json(base/'split_manifest_add.json',{'rows':len(df),'cv_rows':len(cv),'fold_sizes':dict(zip(map(str,range(1,6)),map(int,sizes))),
        'group_leakage':False,'cv_excluded_review_ids':df.loc[df.fold.eq(0),'review_id'].tolist(),'near_duplicate_pairs':near,
        'dataset_sha256':sha(source),'split_sha256':sha(target),'seed':cfg['seed'],'n_splits':5})
    dist=[{'fold':f,'aspect':a,'label':l,'count':int(df.loc[df.fold.eq(f),a].eq(l).sum())} for f in range(1,6) for a,l in cols]
    pd.DataFrame(dist).to_csv(base/'fold_distribution_add.csv',index=False,encoding='utf-8-sig')
    print('Kích thước năm phần:',sizes.astype(int).tolist(),flush=True)


def dataset(root,cfg):
    base=root/cfg['data_dir'];df=read_csv(base/'absa_all_add.csv');f=read_csv(base/'fold_assignments_add.csv')
    manifest=json.loads((base/'split_manifest_add.json').read_text(encoding='utf-8'))
    if sha(base/'absa_all_add.csv')!=manifest['dataset_sha256'] or sha(base/'fold_assignments_add.csv')!=manifest['split_sha256']:
        raise ValueError('Dữ liệu/chia tập đã đổi sau khóa.')
    if set(df.review_id)!=set(f.review_id) or f.review_id.duplicated().any():raise ValueError('Danh sách chia tập không khớp.')
    df=df.merge(f[['review_id','split_group_id','fold']],on='review_id',validate='one_to_one');df['fold']=df.fold.astype(int)
    if df.groupby('split_group_id').fold.nunique().max()>1:raise ValueError('Rò rỉ nhóm.')
    return df


def evidence_pairs(df):
    pairs=[];ambiguous=[]
    for r in df.to_dict('records'):
        for a in ASPECTS:
            spans=[];bad=False
            for frag in [x.strip() for x in r[a+'_evidence'].split('||') if x.strip()]:
                starts=[m.start() for m in re.finditer(re.escape(frag),r['review_text_nlp'])]
                if len(starts)!=1:
                    ambiguous.append({'review_id':r['review_id'],'aspect':a,'fragment':frag,'occurrences':len(starts)});bad=True;break
                spans.append([starts[0],starts[0]+len(frag)])
            if not bad:pairs.append({'review_id':r['review_id'],'aspect':a,'spans':spans,'fold':int(r['fold']),'label':r[a]})
    return pairs,ambiguous


def features(root,cfg):
    from transformers import AutoTokenizer
    from .evidence_features_add import Segmenter,FeatureBuilder,build_cache
    from .evidence_core_add import write_jsonl,encode_bio,decode_bio,evaluate_spans
    base=root/cfg['data_dir'];df=dataset(root,cfg);source=str(root/cfg['vlsp_encoder'])
    token=AutoTokenizer.from_pretrained(source,use_fast=False,local_files_only=True)
    builder=FeatureBuilder(token,Segmenter(root/cfg['vncorenlp_dir']),cfg['evidence'])
    cache=build_cache(base/'tokens_add.jsonl',[{'review_id':r.review_id,'text':r.review_text_nlp} for r in df.itertuples()],builder)
    pairs,amb=evidence_pairs(df);write_jsonl(base/'evidence_pairs_add.jsonl',pairs)
    save_json(base/'evidence_ambiguous_positions_add.json',amb)
    cap=[]; texts=df.set_index('review_id').review_text_nlp.to_dict()
    for p in pairs:
        text=texts[p['review_id']]
        offsets=cache[p['review_id']]['offsets']
        cap.append({'aspect':p['aspect'],'text':text,'gold_spans':p['spans'],'predicted_spans':decode_bio(offsets,encode_bio(offsets,p['spans']),text)})
    save_json(root/cfg['preparation_report_dir']/'token_boundary_ceiling_add.json',{'NOT_MODEL_SCORE':True,'metrics':evaluate_spans(cap)})
    prompts={a:' '.join(builder.segmenter.words('Khía cạnh '+a+': '+cfg['evidence']['aspects'][a])) for a in ASPECTS}
    segmented=df[['review_id']].copy();segmented['phobert_text']=[' '.join(builder.segmenter.words(t)) for t in df.review_text_nlp]
    segmented.to_csv(base/'segmented_add.csv',index=False,encoding='utf-8-sig')
    save_json(base/'preprocessing_add.json',{'prompts':prompts,'segmenter':builder.segmenter.identity,'data_sha256':sha(base/'absa_all_add.csv'),
          'segmented_sha256':sha(base/'segmented_add.csv'),'tokens_sha256':sha(base/'tokens_add.jsonl'),'pairs_sha256':sha(base/'evidence_pairs_add.jsonl'),
          'evidence_prefix_ids':builder.prefixes,'label_data_fingerprint':'raw_text_and_labels','ambiguous_pairs_excluded':len(amb)})
    print(f'Đã chuẩn bị token cho {len(df)} nhận xét; {len(amb)} cặp có vị trí lặp giữ trong học nhãn, loại khỏi học bằng chứng.',flush=True)
