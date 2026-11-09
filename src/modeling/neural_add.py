"""Huấn luyện nhãn và BIO; kiểm tra chéo cố định số lượt, tiếp tục từ bước cập nhật."""
import contextlib
import gc
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
from .common_add import ASPECTS,LABELS,LABEL2ID,fingerprint,prediction_records,report_predictions
from .data_add import dataset,read_csv,stamp,assert_identity
from .evidence_core_add import make_windows,encode_bio,decode_bio,evaluate_spans,read_jsonl,write_jsonl
from src.training.prepare_foody_absa_add import sha,save_json


def environment(request):
    import torch
    if request=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA chưa hoạt động; chọn đúng Python .venv và kiểm tra driver.')
    device='cuda' if request!='cpu' and torch.cuda.is_available() else 'cpu'
    if device=='cpu' and request!='cpu':raise RuntimeError('Không có CUDA; nếu chủ động chạy CPU hãy dùng --device cpu.')
    return device

def seed_everything(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def precision(device):
    import torch
    return torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device=='cuda' else contextlib.nullcontext()

def load_resources(root,cfg):
    from transformers import AutoTokenizer
    base=root/cfg['data_dir'];prep=json.loads((base/'preprocessing_add.json').read_text(encoding='utf-8'))
    df=dataset(root,cfg)
    for name,k in [('tokens_add.jsonl','tokens_sha256'),('segmented_add.csv','segmented_sha256'),('evidence_pairs_add.jsonl','pairs_sha256')]:
        if sha(base/name)!=prep[k]:raise ValueError(f'{name} đã thay đổi; không dùng cache khác dữ liệu.')
    if prep['data_sha256']!=sha(base/'absa_all_add.csv'):raise ValueError('Tiền xử lý khác dữ liệu.')
    token=AutoTokenizer.from_pretrained(root/cfg['vlsp_encoder'],use_fast=False,local_files_only=True)
    cache={r['review_id']:r for r in read_jsonl(base/'tokens_add.jsonl')}
    return df,prep,token,cache

def initialize(root,cfg,kind,token_task):
    from transformers import AutoConfig,AutoModel,AutoModelForSequenceClassification,AutoModelForTokenClassification
    source=str(root/cfg['vlsp_encoder']) if kind in ['phobert_vlsp','evidence'] else cfg['base_encoder']
    if kind in ['phobert_vlsp','evidence']:
        info=json.loads((Path(source)/'source_training_manifest.json').read_text(encoding='utf-8'))
        if not info.get('training_completed') or info.get('test_used') or info.get('transfer_component')!='encoder_only' or info.get('model_name')!=cfg['base_encoder']:
            raise ValueError('Nguồn VLSP chưa hoàn tất hoặc sai quy tắc chuyển giao.')
        for n,digest in info['checkpoint_sha256'].items():
            if sha(Path(source)/n)!=digest:raise ValueError(f'Checkpoint VLSP đã đổi: {n}')
    config=AutoConfig.from_pretrained(source);labels=['O','B','I'] if token_task else LABELS
    config.num_labels=len(labels);config.id2label=dict(enumerate(labels));config.label2id={s:i for i,s in enumerate(labels)}
    config.use_cache=False
    cls=AutoModelForTokenClassification if token_task else AutoModelForSequenceClassification
    model=cls.from_config(config)
    # Chỉ chuyển encoder; đầu nhãn/BIO mới. Không khởi tạo CV từ mô hình đã học toàn bộ Foody.
    source_model=AutoModel.from_pretrained(source,add_pooling_layer=False)
    model.base_model.load_state_dict(source_model.state_dict(),strict=True)
    del source_model;gc.collect();model.gradient_checkpointing_enable()
    return model

def make_examples(df,prep,token,cache,root,cfg,task):
    if task=='evidence':
        pairs=read_jsonl(root/cfg['data_dir']/'evidence_pairs_add.jsonl');examples=[]
        for pair in pairs:
            c=cache[pair['review_id']]
            for w in make_windows(c['ids'],prep['evidence_prefix_ids'][pair['aspect']],encode_bio(c['offsets'],pair['spans']),
                                  cfg['max_length'],cfg['overlap'],token.cls_token_id,token.sep_token_id):
                examples.append({**w,'review_id':pair['review_id'],'fold':pair['fold'],'aspect':pair['aspect']})
        return examples,pairs
    segmented=read_csv(root/cfg['data_dir']/'segmented_add.csv').set_index('review_id').phobert_text.to_dict()
    prompts={a:token.encode(s,add_special_tokens=False) for a,s in prep['prompts'].items()};examples=[]
    for r in df.itertuples():
        ids=token.encode(segmented[r.review_id],add_special_tokens=False)
        for ai,a in enumerate(ASPECTS):
            windows=make_windows(ids,prompts[a],[0]*len(ids),cfg['max_length'],cfg['overlap'],token.cls_token_id,token.sep_token_id)
            if not windows:raise ValueError(f'Không có token: {r.review_id}')
            examples.append({'review_id':r.review_id,'fold':r.fold,'aspect':a,'aspect_idx':ai,'label':LABEL2ID[getattr(r,a)],'windows':windows})
    return examples,None


def collate(items,token,task):
    import torch
    windows=items if task=='evidence' else [w for item in items for w in item['windows']]
    batch=token.pad([{'input_ids':w['input_ids'],'attention_mask':[1]*len(w['input_ids'])} for w in windows],padding=True,return_tensors='pt')
    if task=='evidence':
        n=batch['input_ids'].shape[1];batch['labels']=torch.tensor([w['labels']+[-100]*(n-len(w['labels'])) for w in windows])
    else:batch['owner']=torch.tensor([i for i,item in enumerate(items) for _ in item['windows']])
    return batch

def class_logits(model,batch,n_items,device):
    import torch
    inputs={k:v.to(device) for k,v in batch.items() if k!='owner'};values=model(**inputs).logits.float();owner=batch['owner'].to(device)
    logits=torch.zeros((n_items,5),device=device,dtype=values.dtype).index_add(0,owner,values)
    counts=torch.bincount(owner,minlength=n_items).clamp_min(1)
    return logits/counts[:,None]


def fit(model,examples,token,task,cfg,directory,identity,device):
    import torch
    from transformers import get_linear_schedule_with_warmup
    directory.mkdir(parents=True,exist_ok=True);state_path=directory/'last_checkpoint_add.pt';opts=cfg['evidence'] if task=='evidence' else cfg['phobert']
    batch_size=opts['batch_size'];acc=opts['accumulation_steps'];epochs=opts['epochs']
    n_batches=math.ceil(len(examples)/batch_size);nsteps=math.ceil(n_batches/acc)*epochs
    if not examples:raise ValueError('Tập học rỗng.')
    counts=np.zeros(3 if task=='evidence' else (7,5),dtype=float)
    for x in examples:
        if task=='evidence':
            for y in x['labels']:
                if y>=0:counts[y]+=1
        else:counts[x['aspect_idx'],x['label']]+=1
    if task=='evidence':weights=np.clip(np.sqrt(counts.sum()/(3*np.maximum(counts,1))),.25,5)
    else:
        weights=np.ones((7,5))
        for ai in range(7):
            present=counts[ai]>0;weights[ai,present]=np.clip(np.sqrt(counts[ai].sum()/(present.sum()*counts[ai,present])),.25,4)
    save_json(directory/'training_counts_add.json',{'counts':counts.tolist(),'weights':weights.tolist()})
    model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=opts['learning_rate'],weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(opt,int(.1*nsteps),nsteps)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and not torch.cuda.is_bf16_supported())
    progress={'epoch':0,'next_batch':0,'step':0,'loss_sum':0.0};history=[]
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('Checkpoint khác dữ liệu/cấu hình.')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for st in opt.state.values():
            for k,v in st.items():
                if torch.is_tensor(v) and k!='step':st[k]=v.to(device)
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler']);progress=state['progress'];history=state['history']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if device=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state;print(f"Tiếp tục epoch {progress['epoch']+1}, batch {progress['next_batch']}.",flush=True)
    wt=torch.tensor(weights,device=device,dtype=torch.float32)
    def save():
        temp=state_path.with_suffix('.tmp')
        torch.save({'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'progress':progress,'history':history,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None},temp);temp.replace(state_path)
    if not state_path.exists():save()
    model.train();opt.zero_grad(set_to_none=True)
    while progress['epoch']<epochs:
        ep=progress['epoch'];order=np.random.default_rng(cfg['seed']+ep).permutation(len(examples));start=progress['next_batch']
        for bi in range(start,n_batches):
            items=[examples[int(i)] for i in order[bi*batch_size:(bi+1)*batch_size]];batch=collate(items,token,task)
            divisor=min(acc,n_batches-(bi//acc)*acc)
            with precision(device):
                if task=='evidence':
                    labels=batch.pop('labels').to(device);logits=model(**{k:v.to(device) for k,v in batch.items()}).logits.float()
                    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,3),labels.reshape(-1),weight=wt,ignore_index=-100)
                else:
                    logits=class_logits(model,batch,len(items),device);targets=torch.tensor([x['label'] for x in items],device=device)
                    aspects=torch.tensor([x['aspect_idx'] for x in items],device=device)
                    loss=(torch.nn.functional.cross_entropy(logits,targets,reduction='none')*wt[aspects,targets]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Loss không hữu hạn; dừng và giữ checkpoint trước.')
            scaler.scale(loss/divisor).backward();progress['loss_sum']+=float(loss.detach().cpu())
            if (bi+1)%acc==0 or bi+1==n_batches:
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);old_scale=scaler.get_scale()
                scaler.step(opt);scaler.update()
                if scaler.get_scale()>=old_scale:scheduler.step()
                opt.zero_grad(set_to_none=True);progress['step']+=1;progress['next_batch']=bi+1
                if progress['step']%cfg['save_every_steps']==0:save()
            if (bi+1)%cfg['log_every_batches']==0:print(f"Epoch {ep+1}/{epochs}, batch {bi+1}/{n_batches}, loss={progress['loss_sum']/(bi+1):.4f}",flush=True)
        history.append({'epoch':ep+1,'train_loss':progress['loss_sum']/n_batches,'test_used_for_selection':False})
        progress.update(epoch=ep+1,next_batch=0,loss_sum=0.0);save();save_json(directory/'history_add.json',history)
    return model


def predict_class(model,examples,token,cfg,device):
    import torch
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(examples),cfg['phobert']['eval_batch_size']):
            items=examples[start:start+cfg['phobert']['eval_batch_size']]
            with precision(device):values=class_logits(model,collate(items,token,'class'),len(items),device).float().softmax(-1).cpu().numpy()
            for x,p in zip(items,values):result.append({'review_id':x['review_id'],'aspect':x['aspect'],'probabilities':p})
    return result


def predict_spans(model,pairs,token,cache,prep,cfg,texts,device):
    import torch
    model.eval();result=[]
    with torch.inference_mode():
        for pair in pairs:
            c=cache[pair['review_id']];windows=make_windows(c['ids'],prep['evidence_prefix_ids'][pair['aspect']],[0]*len(c['ids']),cfg['max_length'],cfg['overlap'],token.cls_token_id,token.sep_token_id)
            sums=np.zeros((len(c['ids']),3),dtype=np.float32);counts=np.zeros(len(c['ids']))
            for start in range(0,len(windows),cfg['evidence']['eval_batch_size']):
                group=windows[start:start+cfg['evidence']['eval_batch_size']];batch=collate(group,token,'evidence');batch.pop('labels')
                with precision(device):values=model(**{k:v.to(device) for k,v in batch.items()}).logits.float().cpu().numpy()
                for j,w in enumerate(group):
                    sums[w['start']:w['end']]+=values[j,w['prefix_length']:w['prefix_length']+w['end']-w['start']];counts[w['start']:w['end']]+=1
            if not np.all(counts>0):raise ValueError('Token chưa được bao phủ.')
            mean=sums/counts[:,None];prob=np.exp(mean-mean.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
            text=texts[pair['review_id']];spans=decode_bio(c['offsets'],mean.argmax(-1).tolist(),text)
            scores=[]
            for s,e in spans:
                indices=[i for i,(a,b) in enumerate(c['offsets']) if a<e and b>s and b>a]
                scores.append(float(prob[indices,1:].sum(-1).mean()) if indices else 0.)
            selected = sorted(zip(spans, scores), key=lambda x: x[0][0])
            result.append({'review_id':pair['review_id'],'aspect':pair['aspect'],'text':text,'gold_spans':pair.get('spans',[]),
                'predicted_spans':[x[0] for x in selected],'span_scores':[x[1] for x in selected]})
    return result


def run(root,cfg,kind,requested_device='auto',fold_request='all',final=False):
    import torch
    task='evidence' if kind=='evidence' else 'class';device=environment(requested_device)
    df,prep,token,cache=load_resources(root,cfg);examples,pairs=make_examples(df,prep,token,cache,root,cfg,task)
    out=root/cfg['report_dir']/(kind+'_add');out.mkdir(parents=True,exist_ok=True);identity=stamp(root,cfg)
    # Bind encoder bytes too, so changed source weights cannot reuse completed results.
    enc=root/cfg['vlsp_encoder'];identity['encoder_sha256']={p.name:sha(p) for p in sorted(enc.glob('*')) if p.is_file()}
    assert_identity(out/'run_identity_add.json',identity)
    if final and not (out/'metrics_summary_add.json').exists():raise ValueError(f'Chưa đủ 5 phần đánh giá {kind}.')
    folds=[0] if final else (list(range(1,6)) if fold_request=='all' else [int(fold_request)])
    for f in folds:
        fd=out/('refit_add' if final else f'fold_{f}_add');fd.mkdir(parents=True,exist_ok=True)
        done=fd/'complete_add.json'
        if done.exists():
            if final:
                md=root/('models/phobert_evidence_add' if task=='evidence' else 'models/phobert_foody_add')
                if not (md/'training_complete.json').exists():raise ValueError(f'{md} đã bị dọn nhưng refit ghi hoàn tất. Bỏ báo cáo refit_add và học lại có chủ ý.')
                completion=json.loads((md/'training_complete.json').read_text(encoding='utf-8'))
                if any(not (md/n).is_file() or sha(md/n)!=h for n,h in completion['files_sha256'].items()):raise ValueError(f'{md} thiếu/đổi tệp mô hình.')
            print(f'{kind}: bỏ qua phần đã xong {f}.',flush=True);continue
        train=[x for x in examples if final or (x['fold']>0 and x['fold']!=f)]
        test=[x for x in examples if x['fold']==f] if not final else []
        seed_everything(cfg['seed']+f);model=initialize(root,cfg,kind,task=='evidence')
        model=fit(model,train,token,task,cfg,fd,{'run':identity,'fold':f,'kind':kind},device)
        if not final:
            if task=='class':
                values=predict_class(model,test,token,cfg,device);records=[];lookup=df.set_index('review_id')
                for x in values:
                    r=lookup.loc[[x['review_id']]].reset_index();records+=prediction_records(r,x['aspect'],x['probabilities'][None,:],kind+'_add',f)
                pd.DataFrame(records).to_csv(fd/'predictions_add.csv',index=False,encoding='utf-8-sig')
            else:
                tp=[p for p in pairs if p['fold']==f];texts=df.set_index('review_id').review_text_nlp.to_dict()
                result=predict_spans(model,tp,token,cache,prep,cfg,texts,device)
                write_jsonl(fd/'predictions_add.jsonl',result);save_json(fd/'metrics_add.json',evaluate_spans(result))
        if final or cfg['save_fold_models']:
            destination=root/('models/phobert_evidence_add' if task=='evidence' else 'models/phobert_foody_add') if final else root/f'models/{kind}_add/fold_{f}_add'
            destination.mkdir(parents=True,exist_ok=True);model.save_pretrained(destination,safe_serialization=True);token.save_pretrained(destination)
            if final:export_metadata(destination,df,cfg,prep,task,root,identity)
        save_json(done,{'fold':f,'final_refit':final,'train_units':len(train),'evaluation_performed':not final})
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    if not final:
        required=[out/f'fold_{f}_add/complete_add.json' for f in range(1,6)]
        if all(p.exists() for p in required):
            if task=='class':
                pred=pd.concat([read_csv(p.parent/'predictions_add.csv') for p in required],ignore_index=True);pred['fold']=pred.fold.astype(int)
                report_predictions(pred,out,df[df.fold.gt(0)])
            else:
                result=[r for p in required for r in read_jsonl(p.parent/'predictions_add.jsonl')]
                metrics=evaluate_spans(result)
                per_fold=[json.loads((p.parent/'metrics_add.json').read_text(encoding='utf-8')) for p in required]
                metrics['fold_summary']={k:{'mean':float(np.mean([m['overall'][k] for m in per_fold])), 'std':float(np.std([m['overall'][k] for m in per_fold],ddof=1))} for k in ['span_f1','char_f1','positive_exact_pair_rate']}
                save_json(out/'metrics_summary_add.json',metrics);write_jsonl(out/'out_of_fold_predictions_add.jsonl',result)
                error=[{'review_id':r['review_id'],'aspect':r['aspect'],'review_text_nlp':r['text'],
                    'gold_evidence':' || '.join(r['text'][s:e] for s,e in r['gold_spans']),
                    'predicted_evidence':' || '.join(r['text'][s:e] for s,e in r['predicted_spans'])} for r in result if r['gold_spans']!=r['predicted_spans']]
                pd.DataFrame(error,columns=['review_id','aspect','review_text_nlp','gold_evidence','predicted_evidence']).to_csv(out/'prediction_errors_add.csv',index=False,encoding='utf-8-sig')


def export_metadata(destination,df,cfg,prep,task,root,identity):
    # Tên chuẩn dưới đây là giao thức của Transformers/bộ đọc system cũ, không phải phiên bản dữ liệu.
    df[['review_id','text_group_id']].to_csv(destination/'training_review_ids.csv',index=False,encoding='utf-8-sig')
    save_json(destination/'preprocessing.json',{'word_segmentation_applied':True,'normalization':'NFC','prompts':prep['prompts'],
              'segmenter_resource_sha256':prep['segmenter'],'segmenter':'VnCoreNLP RDRSegmenter'})
    if task=='class':
        save_json(destination/'label_mapping.json',{'labels':LABELS,'label2id':LABEL2ID,'aspects':ASPECTS})
        save_json(destination/'absa_config.json',{'model_version':'phobert_foody_add','labels':LABELS,'aspects':ASPECTS,'prompts':prep['prompts'],
            'max_length':cfg['max_length'],'overlap':cfg['overlap'],'pooling':'mean_logits_over_all_review_windows','training_dataset_sha256':fingerprint(df),
            'A8_trained':False,'evidence_trained':False,'classifier_transferred':False})
    else:save_json(destination/'evidence_config_add.json',{'model_version':'phobert_evidence_add','config':cfg['evidence'],'prefixes':prep['evidence_prefix_ids'],
        'max_length':cfg['max_length'],'overlap':cfg['overlap'],'training_data_sha256':identity['data_sha256'],'A8_trained':False})
    hashes={p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='training_complete.json'}
    save_json(destination/'training_complete.json',{'training_completed':True,'model_version':'phobert_evidence_add' if task=='evidence' else 'phobert_foody_add',
        'train_reviews':len(df),'evaluation_performed':False,'evaluation_belongs_to':'CV models before full refit','files_sha256':hashes})
