"""Dự đoán nhãn/bằng chứng máy và giữ riêng nhãn/bằng chứng tay; xuất đủ xác suất."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

ASPECTS = [f'A{i}' for i in range(1, 8)]
LABELS = ['NONE', 'POS', 'NEU', 'NEG', 'MIX']
FORMAT = 3
EVIDENCE_FIELDS = ['evidence_text','evidence_source','evidence_model_version',
                   'evidence_spans_json','evidence_status','evidence_text_sha256']
SOURCE_EVIDENCE_COLUMNS = {
    source:{key:(f'{source}_evidence_version' if key=='evidence_model_version' else f'{source}_{key}')
            for key in EVIDENCE_FIELDS}
    for source in ['model','manual']
}
MACHINE_FIELDS = ['review_id','aspect','predicted_label',*[f'p_{x}' for x in LABELS],
                  'confidence','margin','model_version','label_source','has_manual_label',
                  'used_for_training','excluded_vlsp_overlap','window_count','review_text_sha256',*EVIDENCE_FIELDS]
COMBINED_FIELDS = ['review_id','aspect','predicted_label','label_source','label_verified','manual_label',
                   'model_label','model_version',*[f'p_{x}' for x in LABELS],'model_confidence','model_margin',
                   'confidence','margin','review_text_sha256','evidence_text','evidence_source',
                   'evidence_model_version','evidence_spans_json','evidence_status','evidence_text_sha256',
                   *SOURCE_EVIDENCE_COLUMNS['model'].values(),*SOURCE_EVIDENCE_COLUMNS['manual'].values()]
MACHINE_NAME = 'aspect_predictions_model_add.csv'
COMBINED_NAME = 'aspect_predictions_with_evidence_add.csv'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def text_hash(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path);temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def write_csv(path, fields, rows):
    path = Path(path);temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)
    temp.replace(path)


def publish_results(work, destination, done):
    """Xuất đúng đường dẫn sổ cũ đọc, sau khi hai bảng đã kiểm tra hoàn tất.

    Giữ bản đã kiểm tra trong thư mục làm việc để có thể xuất tiếp nếu tiến trình
    bị ngắt giữa hai lần thay tệp. Không cần sửa câu lệnh hoặc ô đọc trong sổ.
    """
    expected={MACHINE_NAME,COMBINED_NAME}
    require(set(done['output_sha256'])==expected,'Hồ sơ hoàn tất chưa đủ hai bảng kết quả.')
    destination.mkdir(parents=True,exist_ok=True)
    for name in [MACHINE_NAME,COMBINED_NAME]:
        source=work/name;target=destination/name;digest=done['output_sha256'][name]
        require(source.is_file() and sha(source)==digest,'Bảng đã kiểm tra bị thiếu/đổi: '+str(source))
        if target.is_file() and sha(target)==digest:continue
        temp=target.with_suffix(target.suffix+'.tmp')
        shutil.copyfile(source,temp)
        require(sha(temp)==digest,'Bảng sao chép chưa khớp mã băm: '+str(target))
        temp.replace(target)
    # Ghi hồ sơ hoàn tất sau cùng; lần chạy tiếp có thể xuất lại từ work.
    save_json(destination/'prediction_identity_add.json',done['identity'])
    save_json(destination/'prediction_complete_add.json',done)


def require(value,message):
    if not value:raise ValueError(message)


@contextlib.contextmanager
def output_lock(directory):
    """Khóa hệ điều hành tự nhả khi tiến trình dừng; cho phép tiếp tục sau gián đoạn."""
    stream=(directory/'.prediction.lock').open('a+b')
    try:
        stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError('Đang có tiến trình dự đoán dùng cùng thư mục.') from e
        yield
    finally:stream.close()


def validate_bundle(directory, version, required):
    done=read_json(directory/'training_complete.json')
    require(done.get('training_completed') is True and done.get('model_version')==version,'Sai phiên hoặc mô hình chưa học xong.')
    hashes=done.get('files_sha256',{})
    require(set(required)<=set(hashes),'Hồ sơ mô hình thiếu tệp bắt buộc.')
    for name,digest in hashes.items():
        p=(directory/name).resolve()
        require(p.is_relative_to(directory.resolve()) and p.is_file() and sha(p)==digest,'Tệp mô hình thiếu/đã đổi: '+str(p))
    return sha(directory/'training_complete.json')


def prepare_machine(review_id,aspect,probabilities,version,has_manual,used_for_training,excluded,windows,text):
    values=[float(p) for p in probabilities]
    require(len(values)==5 and all(math.isfinite(p) and 0<=p<=1 for p in values)
            and abs(sum(values)-1)<1e-5,'Phân bố xác suất mô hình không hợp lệ.')
    winner=max(range(5),key=values.__getitem__);ordered=sorted(values,reverse=True)
    return dict(review_id=review_id,aspect=aspect,predicted_label=LABELS[winner],
                **{f'p_{x}':values[i] for i,x in enumerate(LABELS)},confidence=values[winner],
                margin=ordered[0]-ordered[1],model_version=version,label_source='model',
                has_manual_label=str(has_manual).lower(),used_for_training=str(used_for_training).lower(),
                excluded_vlsp_overlap=str(excluded).lower(),window_count=windows,review_text_sha256=text_hash(text))


def validate_machine(row):
    require(row['aspect'] in ASPECTS and row['label_source']=='model','Sai khía cạnh/nguồn nhãn máy.')
    p=[float(row['p_'+x]) for x in LABELS]
    require(all(math.isfinite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<1e-5,'Xác suất sai tổng/miền giá trị.')
    order=sorted(p,reverse=True)
    require(row['predicted_label']==LABELS[max(range(5),key=p.__getitem__)],'Nhãn máy khác phân bố xác suất.')
    require(abs(float(row['confidence'])-order[0])<1e-6
            and abs(float(row['margin'])-(order[0]-order[1]))<1e-6,'confidence/margin không khớp xác suất.')


def evidence_fields(text,label,evidence,spans,source,version='',status=None):
    for s in spans:
        require(type(s['start']) is int and type(s['end']) is int and 0<=s['start']<s['end']<=len(text)
                and text[s['start']:s['end']]==s['text'],'Bằng chứng không khớp vị trí nguyên văn.')
    fragments=[s.strip() for s in evidence.split('||') if s.strip()]
    require(all(s in text for s in fragments),'Bằng chứng không nguyên văn.')
    require(label!='NONE' or not fragments and not spans,'Nhãn NONE chứa bằng chứng.')
    if source=='model':require(evidence==' || '.join(s['text'] for s in spans),'Chuỗi bằng chứng máy không khớp vị trí.')
    return {'evidence_text':evidence,'evidence_source':source,'evidence_model_version':version,
            'evidence_spans_json':json.dumps(spans,ensure_ascii=False),
            'evidence_status':status or ('not_mentioned' if label=='NONE' else 'needs_review_no_evidence' if not spans else
                                       'mixed_evidence_polarity_not_separated' if label=='MIX' else 'model_extracted_unverified'),
            'evidence_text_sha256':text_hash(text)}


def source_evidence(row,source):
    """Đọc các cột bằng chứng độc lập của một nguồn trong bảng tổng hợp."""
    return {key:row[column] for key,column in SOURCE_EVIDENCE_COLUMNS[source].items()}


def tag_evidence(evidence,source):
    return {column:evidence[key] for key,column in SOURCE_EVIDENCE_COLUMNS[source].items()}


def manual_evidence(text,gold,aspect):
    evidence=gold[aspect+'_evidence'];spans=[];ambiguous=False
    for fragment in [s.strip() for s in evidence.split('||') if s.strip()]:
        start=text.find(fragment)
        require(start>=0,'Bằng chứng tay không thuộc văn bản.')
        if text.find(fragment,start+1)>=0:ambiguous=True
        else:spans.append({'start':start,'end':start+len(fragment),'text':fragment})
    if ambiguous:spans=[]
    return evidence_fields(text,gold[aspect],evidence,spans,'manual',status='not_mentioned' if gold[aspect]=='NONE' else
                           'manual_evidence_ambiguous_position' if ambiguous else 'manual_reviewed')


def combine(machine,text,gold=None):
    """Giữ cả hai nguồn độc lập; các cột cũ chứa bản ưu tiên để sổ cũ đọc."""
    result={k:machine[k] for k in ['review_id','aspect','model_version',*[f'p_{x}' for x in LABELS],'review_text_sha256']}
    result.update(model_label=machine['predicted_label'],model_confidence=machine['confidence'],model_margin=machine['margin'])
    result.update(tag_evidence(machine,'model'))
    if gold is not None:
        label=gold[machine['aspect']];confirmed=manual_evidence(text,gold,machine['aspect'])
        result.update(predicted_label=label,manual_label=label,label_source='manual',label_verified='true',confidence='',margin='')
        result.update(tag_evidence(confirmed,'manual'))
        result.update(confirmed)
    else:
        label=machine['predicted_label']
        result.update(predicted_label=label,manual_label='',label_source='model',label_verified='false',
                      confidence=machine['confidence'],margin=machine['margin'])
        result.update({column:'' for column in SOURCE_EVIDENCE_COLUMNS['manual'].values()})
        result.update({key:machine[key] for key in EVIDENCE_FIELDS})
    return result


def validate_pair(machine,combined,text,gold):
    validate_machine(machine)
    require(machine['review_id']==combined['review_id'] and machine['aspect']==combined['aspect'],'Hai bảng lệch cặp nhận xét–khía cạnh.')
    require(machine['predicted_label']==combined['model_label'] and machine['model_version']==combined['model_version'],'Hai bảng lệch nhãn/phiên máy.')
    require(machine['review_text_sha256']==combined['review_text_sha256']==text_hash(text),'Sai liên kết văn bản.')
    for key in [f'p_{x}' for x in LABELS]:require(abs(float(machine[key])-float(combined[key]))<1e-12,'Hai bảng lệch xác suất.')
    require(abs(float(combined['model_confidence'])-float(machine['confidence']))<1e-12
            and abs(float(combined['model_margin'])-float(machine['margin']))<1e-12,'Hai bảng lệch điểm máy.')
    model_evidence=source_evidence(combined,'model')
    require(all(machine[key]==model_evidence[key] for key in EVIDENCE_FIELDS),'Hai bảng lệch bằng chứng máy.')
    require(model_evidence['evidence_source']=='model' and model_evidence['evidence_model_version']
            and not model_evidence['evidence_status'].startswith('manual'),'Bằng chứng máy bị trộn nguồn tay.')
    checked=evidence_fields(text,machine['predicted_label'],model_evidence['evidence_text'],
                            json.loads(model_evidence['evidence_spans_json']),'model',
                            model_evidence['evidence_model_version'],model_evidence['evidence_status'])
    require(checked['evidence_text_sha256']==model_evidence['evidence_text_sha256'],'Sai liên kết văn bản bằng chứng máy.')
    if gold is not None:
        require(combined['label_source']=='manual' and combined['predicted_label']==gold[machine['aspect']]
                and combined['manual_label']==gold[machine['aspect']] and combined['confidence']==combined['margin']=='',
                'Nhãn tay hoặc điểm của nguồn nhãn bị ghi đè.')
        require(combined['label_verified']=='true' and combined['evidence_source']=='manual','Sai trạng thái/nguồn xác nhận tay.')
        require(combined['evidence_text']==gold[machine['aspect']+'_evidence'],'Bằng chứng tay bị thay đổi.')
        confirmed=source_evidence(combined,'manual')
        require(confirmed==manual_evidence(text,gold,machine['aspect']),'Bằng chứng tay độc lập không khớp nguồn gán tay.')
        require(all(combined[key]==confirmed[key] for key in EVIDENCE_FIELDS),'Bằng chứng ưu tiên không giữ bản tay.')
    else:
        require(combined['label_source']=='model' and combined['predicted_label']==machine['predicted_label'] and combined['manual_label']=='',
                'Nguồn nhãn máy không khớp.')
        require(combined['evidence_source']=='model' and abs(float(combined['confidence'])-float(machine['confidence']))<1e-12
                and abs(float(combined['margin'])-float(machine['margin']))<1e-12,'Sai nguồn bằng chứng hoặc điểm máy.')
        require(all(combined[column]=='' for column in SOURCE_EVIDENCE_COLUMNS['manual'].values()),'Tự tạo bằng chứng tay cho nhận xét chưa gán.')
        require(all(combined[key]==model_evidence[key] for key in EVIDENCE_FIELDS),'Bằng chứng ưu tiên khác bản máy.')
    checked=evidence_fields(text,combined['predicted_label'],combined['evidence_text'],json.loads(combined['evidence_spans_json']),
                            combined['evidence_source'],combined['evidence_model_version'],combined['evidence_status'])
    require(checked['evidence_text_sha256']==combined['evidence_text_sha256'],'Sai liên kết văn bản bằng chứng.')


def verify_tables(machine_path,combined_path,source,gold):
    """Đối chiếu đầy đủ từng cặp; điểm p luôn thuộc model_label kể cả khi hiển thị nhãn tay."""
    from itertools import zip_longest
    seen=set();manual_pairs=0;disagreements=0;versions=set();evidence_versions=set()
    model_evidence_pairs=0;manual_evidence_pairs=0
    for m,c in zip_longest(csv_rows(machine_path),csv_rows(combined_path)):
        require(m is not None and c is not None,'Hai bảng thiếu/thừa dòng.')
        rid=m['review_id'];key=rid,m['aspect']
        require(rid in source and key not in seen,'Trùng cặp hoặc mã ngoài corpus.')
        seen.add(key);versions.add(m['model_version']);g=gold.get(rid)
        validate_pair(m,c,source[rid]['review_text_nlp'],g)
        evidence_versions.add(m['evidence_model_version'])
        model_evidence_pairs+=bool(m['evidence_text'])
        manual_evidence_pairs+=bool(c['manual_evidence_text'])
        manual_pairs+=g is not None
        disagreements+=g is not None and m['predicted_label']!=g[m['aspect']]
    require(len(seen)==len(source)*7,'Chưa đủ bảy khía cạnh cho mỗi nhận xét.')
    require(len(versions)==1,'Hai bảng đang trộn phiên mô hình.')
    require(len(evidence_versions)==1,'Hai bảng đang trộn phiên mô hình bằng chứng.')
    return {'reviews':len(source),'rows':len(seen),'manual_pairs':manual_pairs,
            'manual_model_disagreements':disagreements,'independent_model_evaluation':False,
            'model_evidence_nonempty_pairs':model_evidence_pairs,'manual_evidence_nonempty_pairs':manual_evidence_pairs}


def old_evidence(base,source,model_signatures,corpus_sha):
    """Chỉ dùng lại bằng chứng máy khi hồ sơ nguồn khớp corpus và cả hai mô hình."""
    path=base/COMBINED_NAME;proof=base/'prediction_complete_add.json'
    if not path.is_file() or not proof.is_file():return {},None
    done=read_json(proof)
    require(done['sha256']==sha(path),'Bảng bằng chứng cũ khác hồ sơ; không dùng lại.')
    identity=done['identity']
    if identity.get('corpus_sha256')!=corpus_sha or identity.get('model_signatures')!=model_signatures:return {},None
    keep={}
    for row in csv_rows(path):
        rid=row['review_id']
        if rid not in source:continue
        if 'model_evidence_text' in row:
            # Bản mới giữ máy riêng cả khi cột ưu tiên đang dùng nhãn tay.
            raw=source_evidence(row,'model')
            row={**row,**raw,'predicted_label':row['model_label'],'label_source':'model'}
        elif row['label_source']!='model':
            # Bản cũ chỉ giữ bằng chứng tay: tuyệt đối không coi đó là máy.
            continue
        require(row['evidence_source']=='model' and row['aspect'] in ASPECTS,'Sai nguồn bằng chứng cũ.')
        require(not row['evidence_status'].startswith('manual'),'Không dùng bằng chứng tay làm bằng chứng máy.')
        require(row['model_version']=='phobert_foody_add' and row['evidence_model_version']=='phobert_evidence_add',
                'Bằng chứng cũ khác phiên mô hình khai báo.')
        key=rid,row['aspect'];require(key not in keep,'Bằng chứng cũ trùng cặp.')
        keep[key]=row
    return keep,done['sha256']


def _load_runtime(root,cfg,device,cm,em,absa,ep,out):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification,AutoModelForTokenClassification,AutoTokenizer
    from .evidence_core_add import make_windows,decode_bio
    from .evidence_features_add import FeatureBuilder,Segmenter,build_cache
    tokenizer=AutoTokenizer.from_pretrained(cm,use_fast=False,local_files_only=True)
    segmenter=Segmenter(root/cfg['vncorenlp_dir'])
    prep=read_json(cm/'preprocessing.json')
    require(segmenter.identity==prep['segmenter_resource_sha256'],'Tài nguyên tách từ khác lúc huấn luyện.')
    builder=FeatureBuilder(tokenizer,segmenter,cfg['evidence'])
    require(builder.prefixes==ep['prefixes'],'Lời nhắc bằng chứng khác mô hình đã học.')
    require(absa['pooling']=='mean_logits_over_all_review_windows','Mô hình dùng cách gộp cửa sổ chưa được hỗ trợ.')
    prompts={a:tokenizer.encode(absa['prompts'][a],add_special_tokens=False) for a in ASPECTS}
    classifier=AutoModelForSequenceClassification.from_pretrained(cm,local_files_only=True).to(device)
    classifier.eval();extractor=None;cache={}
    def precision():
        return torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device=='cuda' else contextlib.nullcontext()
    def classify(rows):
        examples=[]
        for row in rows:
            ids=tokenizer.encode(' '.join(segmenter.words(row['review_text_nlp'])),add_special_tokens=False)
            for a in ASPECTS:
                windows=make_windows(ids,prompts[a],[0]*len(ids),cfg['max_length'],cfg['overlap'],tokenizer.cls_token_id,tokenizer.sep_token_id)
                require(windows,'Bộ tách từ không tạo được token cho: '+row['review_id'])
                examples.append({'review_id':row['review_id'],'aspect':a,'windows':windows})
        result=[]
        with torch.inference_mode():
            size=int(cfg['phobert']['eval_batch_size'])
            for start in range(0,len(examples),size):
                items=examples[start:start+size];windows=[w for x in items for w in x['windows']]
                batch=tokenizer.pad([{'input_ids':w['input_ids'],'attention_mask':[1]*len(w['input_ids'])} for w in windows],padding=True,return_tensors='pt')
                owner=torch.tensor([i for i,x in enumerate(items) for _ in x['windows']],device=device)
                with precision():
                    logits=classifier(**{k:v.to(device) for k,v in batch.items()}).logits.float()
                    sums=torch.zeros((len(items),5),device=device,dtype=logits.dtype).index_add(0,owner,logits)
                    values=(sums/torch.bincount(owner,minlength=len(items)).clamp_min(1)[:,None]).softmax(-1).cpu().numpy()
                for x,p in zip(items,values):result.append({**x,'probabilities':p,'window_count':len(x['windows'])})
        return result
    def extract(rows,pairs):
        nonlocal extractor,cache
        if not pairs:return {}
        if extractor is None:
            extractor=AutoModelForTokenClassification.from_pretrained(em,local_files_only=True).to(device);extractor.eval()
        texts={r['review_id']:r['review_text_nlp'] for r in rows}
        selected_ids={p['review_id'] for p in pairs}
        cache=build_cache(out/'tokens_add.jsonl',[{'review_id':rid,'text':texts[rid]} for rid in selected_ids],builder,cache=cache)
        result={}
        with torch.inference_mode():
            for p in pairs:
                c=cache[p['review_id']]
                windows=make_windows(c['ids'],ep['prefixes'][p['aspect']],[0]*len(c['ids']),cfg['max_length'],cfg['overlap'],tokenizer.cls_token_id,tokenizer.sep_token_id)
                sums=np.zeros((len(c['ids']),3),dtype=np.float32);counts=np.zeros(len(c['ids']))
                for start in range(0,len(windows),int(cfg['evidence']['eval_batch_size'])):
                    group=windows[start:start+int(cfg['evidence']['eval_batch_size'])]
                    batch=tokenizer.pad([{'input_ids':w['input_ids'],'attention_mask':[1]*len(w['input_ids'])} for w in group],padding=True,return_tensors='pt')
                    with precision():values=extractor(**{k:v.to(device) for k,v in batch.items()}).logits.float().cpu().numpy()
                    for j,w in enumerate(group):
                        n=w['end']-w['start'];sums[w['start']:w['end']]+=values[j,w['prefix_length']:w['prefix_length']+n];counts[w['start']:w['end']]+=1
                require(np.all(counts>0),'Token bằng chứng chưa được bao phủ.')
                text=texts[p['review_id']];raw=decode_bio(c['offsets'],(sums/counts[:,None]).argmax(-1).tolist(),text)
                spans=[{'start':s,'end':e,'text':text[s:e]} for s,e in raw]
                result[p['review_id'],p['aspect']]=evidence_fields(text,p['predicted_label'],' || '.join(s['text'] for s in spans),spans,'model',ep['model_version'])
        return result
    return classify,extract


def run(root,cfg,requested='auto',limit=0,output_dir=None):
    """Giữ bốn tham số cũ để run_stage('predict', ...) gọi như trước."""
    require(isinstance(limit,int) and limit>=0,'limit phải là số nguyên không âm.')
    root=Path(root).resolve();corpus=root/cfg['corpus_file'];gold_path=root/cfg['data_dir']/'absa_all_add.csv'
    records=[{k:r.get(k,'') for k in ['review_id','review_text_nlp','text_group_id']} for r in csv_rows(corpus)]
    require(records and len({r['review_id'] for r in records})==len(records),'Corpus trống/trùng mã.')
    require(all(r['review_id'].strip() and r['review_text_nlp'].strip() for r in records),'Corpus có mã/văn bản trống.')
    all_ids={r['review_id'] for r in records}
    if limit:records=records[:limit]
    source={r['review_id']:r for r in records};gold={}
    for r in csv_rows(gold_path):
        require(r['review_id'] not in gold,'Nhãn tay trùng mã.')
        require(r.get('uncertain','0') in {'','0','false','False'} and all(r[a] in LABELS for a in ASPECTS),'Nhãn tay chưa xác nhận/hợp lệ.')
        require(r.get('label_status') in {'Hoàn thành','ĐÃ GÁN - ĐÃ RÀ TRƯỚC','DA_DUYET','DA_XAC_NHAN','CONFIRMED'},
                'Nhãn tay chưa có trạng thái xác nhận: '+r['review_id'])
        gold[r['review_id']]=r
    require(set(gold)<=all_ids,'Nhãn tay có mã ngoài corpus.')
    for rid in set(source)&set(gold):require(source[rid]['review_text_nlp']==gold[rid]['review_text_nlp'],'Nhãn tay khác văn bản corpus: '+rid)
    cm=root/'models/phobert_foody_add';em=root/'models/phobert_evidence_add'
    model_signatures={cm.name:validate_bundle(cm,cm.name,['model.safetensors','config.json','absa_config.json','preprocessing.json','training_review_ids.csv']),
                      em.name:validate_bundle(em,em.name,['model.safetensors','config.json','evidence_config_add.json'])}
    absa=read_json(cm/'absa_config.json');ep=read_json(em/'evidence_config_add.json')
    require(absa['aspects']==ASPECTS and absa['labels']==LABELS,'Mô hình khác bộ nhãn/khía cạnh.')
    require(absa['model_version']==cm.name and ep['model_version']==em.name,'Cấu hình khác phiên mô hình.')
    require(read_json(cm/'config.json').get('id2label')=={str(i):label for i,label in enumerate(LABELS)},'Thứ tự nhãn máy khác cấu hình đã học.')
    require(read_json(em/'config.json').get('id2label')=={'0':'O','1':'B','2':'I'},'Bộ trích khác cấu trúc BIO đã học.')
    require(all(absa[k]==cfg[k] and ep[k]==cfg[k] for k in ['max_length','overlap']),'Độ dài cửa sổ khác cấu hình đã học.')
    trained={r['review_id'] for r in csv_rows(cm/'training_review_ids.csv')}
    excluded_path=root/cfg['data_dir']/'source_overlap_audit_add.json'
    excluded=set(read_json(excluded_path).get('excluded_cv_review_ids',[])) if excluded_path.is_file() else set()
    base=root/cfg['prediction_dir'];destination=Path(output_dir) if output_dir else base
    if not destination.is_absolute():destination=root/destination
    # Giữ nguyên thư mục kết quả đầy đủ và chạy thử của bộ mã/sổ cũ.
    if limit:destination=destination/f'smoke_{limit}_add'
    out=destination/'day_du'
    out.mkdir(parents=True,exist_ok=True)
    signature={'format':FORMAT,'corpus_sha256':sha(corpus),'manual_sha256':sha(gold_path),'model_signatures':model_signatures,
               'limit':limit,'chunk_size':int(cfg['prediction_chunk_size']),'program_sha256':sha(__file__),
               'inference':{k:cfg[k] for k in ['max_length','overlap','phobert','evidence','vncorenlp_dir']},
               'features_sha256':sha(Path(__file__).with_name('evidence_features_add.py')),
               'core_sha256':sha(Path(__file__).with_name('evidence_core_add.py')),
               'excluded_cv_sha256':sha(excluded_path) if excluded_path.is_file() else None}
    require(signature['chunk_size']>0,'Kích thước phần dự đoán phải dương.')
    with output_lock(out):
        identity_path=out/'prediction_identity_add.json';complete_path=out/'prediction_complete_add.json'
        if identity_path.exists():require(read_json(identity_path)==signature,'Nguồn/mã/cấu hình đã đổi; dùng --output-dir khác để tạo lần dự đoán mới.')
        else:save_json(identity_path,signature)
        if complete_path.exists():
            done=read_json(complete_path);require(done['identity']==signature,'Hồ sơ hoàn tất khác lần dự đoán.')
            require(all((out/n).is_file() and sha(out/n)==h for n,h in done['output_sha256'].items()),'Kết quả hoàn tất bị thiếu/đổi.')
            publish_results(out,destination,done)
            print('Dự đoán đầy đủ đã hoàn tất:',destination,flush=True);return done
        reusable,reused_sha=old_evidence(base,source,model_signatures,signature['corpus_sha256'])
        import torch
        require(requested!='cuda' or torch.cuda.is_available(),'CUDA chưa hoạt động; chọn đúng môi trường đã huấn luyện.')
        device='cuda' if requested!='cpu' and torch.cuda.is_available() else 'cpu'
        require(requested=='cpu' or device=='cuda','Không có CUDA; dùng --device cpu nếu chủ động chạy CPU.')
        classify=extract=None;parts=out/'parts';parts.mkdir(exist_ok=True);paths=[];reused_count=0
        for index,start in enumerate(range(0,len(records),signature['chunk_size'])):
            chunk=records[start:start+signature['chunk_size']];ids={r['review_id'] for r in chunk}
            mp=parts/f'may_{index:06d}.csv';cp=parts/f'uu_tien_{index:06d}.csv';dp=parts/f'phan_{index:06d}.json';paths.append((mp,cp))
            if dp.exists():
                d=read_json(dp)
                require(d['review_ids']==[r['review_id'] for r in chunk] and d['machine_sha256']==sha(mp) and d['combined_sha256']==sha(cp),'Phần dự đoán đã lưu bị đổi.')
                verify_tables(mp,cp,{rid:source[rid] for rid in ids},gold);reused_count+=d.get('reused_evidence_pairs',0);continue
            if classify is None:classify,extract=_load_runtime(root,cfg,device,cm,em,absa,ep,out)
            predicted=classify(chunk);machine=[];selected_evidence={};needed=[];seen=set();reused_part=0
            for x in predicted:
                rid,a=x['review_id'],x['aspect'];key=rid,a
                require(rid in ids and a in ASPECTS and key not in seen,'Suy luận trả thiếu/trùng/ngoài cặp yêu cầu.');seen.add(key)
                m=prepare_machine(rid,a,x['probabilities'],absa['model_version'],rid in gold,rid in trained,rid in excluded,x['window_count'],source[rid]['review_text_nlp']);machine.append(m)
                old=reusable.get(key)
                if old and old['predicted_label']==m['predicted_label']:
                    selected_evidence[key]=evidence_fields(source[rid]['review_text_nlp'],m['predicted_label'],old['evidence_text'],json.loads(old['evidence_spans_json']),'model',ep['model_version'],old['evidence_status']);reused_part+=1
                elif m['predicted_label']=='NONE':selected_evidence[key]=evidence_fields(source[rid]['review_text_nlp'],'NONE','',[],'model',ep['model_version'])
                else:needed.append({'review_id':rid,'aspect':a,'predicted_label':m['predicted_label']})
            require(len(machine)==len(chunk)*7,'Suy luận không đủ bảy khía cạnh mỗi nhận xét.')
            extracted=extract(chunk,needed)
            require(set(extracted)=={(p['review_id'],p['aspect']) for p in needed},'Bộ trích trả thiếu/thừa cặp bằng chứng.')
            selected_evidence.update(extracted)
            # Mỗi dòng máy giữ bằng chứng máy riêng, kể cả khi đã có nhãn tay.
            for m in machine:m.update(selected_evidence[m['review_id'],m['aspect']])
            combined=[combine(m,source[m['review_id']]['review_text_nlp'],gold.get(m['review_id'])) for m in machine]
            write_csv(mp,MACHINE_FIELDS,machine);write_csv(cp,COMBINED_FIELDS,combined)
            verify_tables(mp,cp,{rid:source[rid] for rid in ids},gold)
            save_json(dp,{'review_ids':[r['review_id'] for r in chunk],'machine_sha256':sha(mp),'combined_sha256':sha(cp),'reused_evidence_pairs':reused_part})
            reused_count+=reused_part;print(f'Đã dự đoán đầy đủ {min(start+len(chunk),len(records))}/{len(records)} nhận xét.',flush=True)
        write_csv(out/MACHINE_NAME,MACHINE_FIELDS,(r for mp,cp in paths for r in csv_rows(mp)))
        write_csv(out/COMBINED_NAME,COMBINED_FIELDS,(r for mp,cp in paths for r in csv_rows(cp)))
        summary=verify_tables(out/MACHINE_NAME,out/COMBINED_NAME,source,gold)
        require(sha(corpus)==signature['corpus_sha256'] and sha(gold_path)==signature['manual_sha256'],'Nguồn đổi trong lúc dự đoán.')
        for md in [cm,em]:require(validate_bundle(md,md.name,['model.safetensors','config.json'])==model_signatures[md.name],'Mô hình đổi trong lúc dự đoán.')
        done={**summary,'sha256':sha(out/COMBINED_NAME),'output_sha256':{n:sha(out/n) for n in [MACHINE_NAME,COMBINED_NAME]},
              'identity':signature,'A8_included':False,'confidence_calibrated':False,'manual_labels_modified':False,
              'manual_evidence_modified':False,'separate_model_and_manual_evidence':True,
              'model_evidence_for_manual_reviews':True,
              'reused_evidence_pairs':reused_count,'reused_evidence_source_sha256':reused_sha,
              'probability_label_column':'model_label','preferred_label_column':'predicted_label',
              'probability_label_columns':{MACHINE_NAME:'predicted_label',COMBINED_NAME:'model_label'},
              'probability_method':'softmax(mean_logits_over_all_review_windows)',
              'probability_labels':LABELS,'confidence_definition':'max_probability',
              'margin_definition':'largest_probability_minus_second_largest',
              'combined_columns':{'model_label':'model_label','model_evidence':'model_evidence_text',
                                  'manual_label':'manual_label','manual_evidence':'manual_evidence_text',
                                  'preferred_label':'predicted_label','preferred_evidence':'evidence_text'}}
        save_json(complete_path,done)
        publish_results(out,destination,done)
        print('Hoàn tất. Nhãn máy:',destination/MACHINE_NAME,'\nBảng ưu tiên nhãn tay:',destination/COMBINED_NAME,flush=True)
        return done


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--output-dir',type=Path,default=None)
    args=p.parse_args(argv);require(args.limit>=0,'limit phải không âm.')
    cfg=read_json(args.project_root/'configs/training_add/training_add.json')
    run(args.project_root,cfg,args.device,args.limit,args.output_dir)


if __name__=='__main__':main()
