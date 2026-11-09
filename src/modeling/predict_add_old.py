"""Dự đoán có chia phần/tiếp tục; tách thử nhỏ khỏi đầu ra toàn corpus."""
import json
from pathlib import Path
import pandas as pd
from .common_add import ASPECTS,LABELS
from .data_add import read_csv,assert_identity
from .neural_add import environment,predict_class,predict_spans
from .evidence_core_add import make_windows
from .evidence_features_add import FeatureBuilder,Segmenter,build_cache
from src.training.prepare_foody_absa_add import sha,save_json


def run(root,cfg,requested='auto',limit=0):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoModelForTokenClassification,AutoTokenizer
    corpus=root/cfg['corpus_file']
    if not corpus.is_file():raise FileNotFoundError(f'Thiếu corpus {corpus}; sửa corpus_file trong configs/training_add/training_add.json.')
    source=read_csv(corpus)
    if not {'review_id','review_text_nlp'}<=set(source):raise ValueError('Corpus phải có review_id, review_text_nlp.')
    if source.review_id.duplicated().any():raise ValueError('review_id corpus trùng.')
    if source.review_text_nlp.str.strip().eq('').any():raise ValueError('Corpus chứa nhận xét trống.')
    if limit:source=source.head(limit)
    out=root/cfg['prediction_dir']
    if limit:out=out/f'smoke_{limit}_add'
    out.mkdir(parents=True,exist_ok=True);device=environment(requested)
    cm=root/'models/phobert_foody_add';em=root/'models/phobert_evidence_add';proof={}
    for md in [cm,em]:
        done=json.loads((md/'training_complete.json').read_text(encoding='utf-8'))
        if not done.get('training_completed'):raise ValueError(f'{md} chưa học xong.')
        for n,digest in done['files_sha256'].items():
            if sha(md/n)!=digest:raise ValueError(f'{md/n} đã thay đổi.')
        proof[md.name]=sha(md/'training_complete.json')
    signature={'corpus_sha256':sha(corpus),'model_signatures':proof,'limit':limit,'chunk_size':cfg['prediction_chunk_size']}
    assert_identity(out/'prediction_identity_add.json',signature)
    if (out/'prediction_complete_add.json').exists():print('Dự đoán này đã hoàn tất.');return
    token=AutoTokenizer.from_pretrained(cm,use_fast=False,local_files_only=True)
    builder=FeatureBuilder(token,Segmenter(root/cfg['vncorenlp_dir']),cfg['evidence'])
    absa=json.loads((cm/'absa_config.json').read_text(encoding='utf-8'))
    ep=json.loads((em/'evidence_config_add.json').read_text(encoding='utf-8'))
    if any(absa[k]!=cfg[k] or ep[k]!=cfg[k] for k in ['max_length','overlap']):raise ValueError('Độ dài cửa sổ khác cấu hình đã học.')
    prompts={a:token.encode(s,add_special_tokens=False) for a,s in absa['prompts'].items()}
    classifier=AutoModelForSequenceClassification.from_pretrained(cm,local_files_only=True).to(device)
    extractor=AutoModelForTokenClassification.from_pretrained(em,local_files_only=True).to(device)
    gold=read_csv(root/cfg['data_dir']/'absa_all_add.csv').set_index('review_id')
    parts=out/'parts_add';parts.mkdir(exist_ok=True);paths=[]
    for index,start in enumerate(range(0,len(source),cfg['prediction_chunk_size'])):
        path=parts/f'part_{index:06d}_add.csv';done=parts/f'part_{index:06d}_add.json';paths.append(path)
        if done.exists():
            if not path.exists() or sha(path)!=json.loads(done.read_text())['sha256']:raise ValueError(f'Phần dự đoán bị đổi {path}')
            continue
        chunk=source.iloc[start:start+cfg['prediction_chunk_size']];rows=chunk.to_dict('records');texts={r['review_id']:r['review_text_nlp'] for r in rows}
        cache=build_cache(out/'tokens_add.jsonl',[{'review_id':rid,'text':t} for rid,t in texts.items()],builder)
        examples=[]
        for r in rows:
            rid=r['review_id'];ids=token.encode(' '.join(builder.segmenter.words(r['review_text_nlp'])),add_special_tokens=False)
            for a in ASPECTS:
                w=make_windows(ids,prompts[a],[0]*len(ids),cfg['max_length'],cfg['overlap'],token.cls_token_id,token.sep_token_id)
                examples.append({'review_id':rid,'aspect':a,'windows':w})
        pred=predict_class(classifier,examples,token,cfg,device);result=[];needed=[]
        for x in pred:
            rid,a=x['review_id'],x['aspect'];p=x['probabilities'];rank=p.argsort()[::-1]
            label=LABELS[int(rank[0])];label_source='model';evidence='';evidence_source='model';confidence=float(p[rank[0]])
            if rid in gold.index:
                original=gold.loc[rid]
                if texts[rid]!=original.review_text_nlp:raise ValueError(f'Corpus khác văn bản gán tay {rid}; đối chiếu lại.')
                label=original[a];evidence=original[a+'_evidence'];label_source='manual';evidence_source='manual';confidence=None
            item={'review_id':rid,'aspect':a,'predicted_label':label,'label_source':label_source,'confidence':confidence,
                 'model_version':'phobert_foody_add','evidence_text':evidence,'evidence_source':evidence_source,
                 'evidence_model_version':'phobert_evidence_add','evidence_spans_json':'[]',
                 'evidence_status':'not_mentioned' if label=='NONE' else 'manual_reviewed' if label_source=='manual' else 'needs_review_no_evidence'}
            if label_source=='manual' and evidence:
                spans=[];ambiguous=False
                for fragment in evidence.split(' || '):
                    s=texts[rid].find(fragment)
                    if s<0 or texts[rid].find(fragment,s+1)>=0:ambiguous=True;continue
                    spans.append({'start':s,'end':s+len(fragment),'text':fragment})
                if ambiguous:item['evidence_status']='manual_evidence_ambiguous_position'
                item['evidence_spans_json']=json.dumps([] if ambiguous else spans,ensure_ascii=False)
            if label!='NONE' and label_source=='model':needed.append({'review_id':rid,'aspect':a})
            result.append(item)
        inferred=predict_spans(extractor,needed,token,cache,{'evidence_prefix_ids':ep['prefixes']},cfg,texts,device)
        lookup={(r['review_id'],r['aspect']):r for r in inferred}
        for item in result:
            match=lookup.get((item['review_id'],item['aspect']))
            if match:
                spans=[{'start':s,'end':e,'text':match['text'][s:e]} for s,e in match['predicted_spans']]
                item['evidence_text']=' || '.join(s['text'] for s in spans);item['evidence_spans_json']=json.dumps(spans,ensure_ascii=False)
                if spans:item['evidence_status']='mixed_evidence_polarity_not_separated' if item['predicted_label']=='MIX' else 'model_extracted_unverified'
        temp=path.with_suffix('.tmp');pd.DataFrame(result).to_csv(temp,index=False,encoding='utf-8-sig');temp.replace(path)
        save_json(done,{'sha256':sha(path),'reviews':len(rows),'rows':len(result)});print(f'Dự đoán {min(start+len(rows),len(source))}/{len(source)}',flush=True)
    target=out/'aspect_predictions_with_evidence_add.csv';temp=target.with_suffix('.tmp');first=True;count=0
    for path in paths:
        block=read_csv(path);block.to_csv(temp,index=False,encoding='utf-8-sig' if first else 'utf-8',mode='w' if first else 'a',header=first);first=False;count+=len(block)
    if count!=len(source)*7:raise ValueError('Đầu ra không đủ bảy khía cạnh mỗi nhận xét.')
    temp.replace(target);save_json(out/'prediction_complete_add.json',{'reviews':len(source),'rows':count,'sha256':sha(target),'identity':signature,'A8_included':False})
