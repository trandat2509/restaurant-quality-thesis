"""Bộ trích bằng chứng duy nhất cho mô hình phobert_evidence_add đang vận hành."""
import json
from pathlib import Path

from src.modeling.phobert_system_common import ASPECTS, location, object_hash
from src.pipeline.prepare_system_data import jread, require, sha, text_hash, atomic_json, csv_rows


def evidence_bundle(cfg, profile):
    require(profile.get('format','foody_add') == 'foody_add' and profile.get('model_dir'),
            'Bộ trích hiện tại cần hồ sơ format=foody_add và model_dir của mô hình mới.')
    directory = location(dict(cfg, model_dir=profile['model_dir']), 'model_dir')
    done = jread(directory / 'training_complete.json')
    require(done.get('training_completed') is True, 'Bộ trích bổ sung chưa học xong.')
    hashes = done.get('files_sha256', {})
    require({'model.safetensors', 'config.json', 'evidence_config_add.json', 'preprocessing.json'} <= set(hashes),
            'Thiếu hồ sơ trọng số bộ trích bổ sung.')
    for name, digest in hashes.items():
        p = (directory / name).resolve()
        require(p.is_relative_to(directory) and p.is_file() and sha(p) == digest, 'Tệp bộ trích thiếu/đã đổi: ' + name)
    ec = jread(directory / 'evidence_config_add.json')
    model_config = jread(directory / 'config.json')
    require(model_config.get('id2label') == {'0':'O','1':'B','2':'I'}, 'Bộ trích khác cấu trúc nhãn BIO đã học.')
    require(ec['model_version'] == done['model_version'] == profile['model_version'], 'Sai phiên bộ trích.')
    require(set(ec['prefixes']) == set(ASPECTS), 'Bộ trích thiếu khía cạnh.')
    identity = {'format': 'foody_add', 'model_version': ec['model_version'], 'model_dir': profile['model_dir'],
                'vncorenlp_dir': profile.get('vncorenlp_dir', cfg['vncorenlp_dir']), 'files_sha256': hashes,
                'completion_sha256': sha(directory / 'training_complete.json')}
    if profile.get('files_sha256'):
        require(profile == identity, 'Bộ trích đã đổi sau đăng ký; tạo phiên mới.')
    return identity


def predict_evidence(cfg, profile, master, pairs, output_path, device='auto'):
    import numpy as np
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from src.modeling.evidence_features_add import FeatureBuilder, Segmenter, build_cache
    from src.modeling.evidence_core_add import make_windows, decode_bio
    from src.pipeline.update_system_data import write_csv
    profile = evidence_bundle(cfg, profile)
    directory = location(dict(cfg, model_dir=profile['model_dir']), 'model_dir')
    ec = jread(directory / 'evidence_config_add.json')
    prep = jread(directory / 'preprocessing.json')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signature = object_hash({'profile': profile, 'pairs': pairs,
                             'texts': {p['review_id']: text_hash(master[p['review_id']]['review_text_nlp']) for p in pairs}})
    proof_path = output_path.with_suffix('.proof.json')
    proof = jread(proof_path) if proof_path.exists() else {'signature': signature, 'parts': {}}
    require(proof['signature'] == signature, 'Đầu vào bộ trích khác lần chạy đang tiếp tục.')
    if proof.get('complete'):
        require(output_path.is_file() and sha(output_path) == proof['output_sha256'], 'Bằng chứng đã lưu bị đổi.')
        return output_path
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA chưa hoạt động.')
    target = 'cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=False, local_files_only=True)
    segmenter = Segmenter(location(dict(cfg, vncorenlp_dir=profile['vncorenlp_dir']), 'vncorenlp_dir'))
    require(segmenter.identity == prep['segmenter_resource_sha256'], 'Tài nguyên tách từ khác lúc huấn luyện.')
    config = dict(ec['config'], max_length=ec['max_length'], overlap=ec['overlap'])
    builder = FeatureBuilder(tokenizer, segmenter, config)
    require(builder.prefixes == ec['prefixes'], 'Lời nhắc bộ trích khác lúc huấn luyện.')
    model = AutoModelForTokenClassification.from_pretrained(directory, local_files_only=True).to(target)
    model.eval()
    fields = ['review_id','aspect','predicted_label','model_version','evidence_text','evidence_spans_json',
              'evidence_status','evidence_source','evidence_model_version','evidence_text_sha256']
    part_dir = output_path.parent / 'evidence_add_parts'
    part_dir.mkdir(exist_ok=True)
    paths = []
    cache = {}
    for start in range(0, len(pairs), 64):
        part = part_dir / f'part_{start//64:05d}.csv'
        paths.append(part)
        if str(start) in proof['parts']:
            require(part.is_file() and sha(part) == proof['parts'][str(start)], 'Phần bằng chứng đã lưu bị đổi.')
            continue
        chunk = pairs[start:start+64]
        reviews = [{'review_id': rid, 'text': master[rid]['review_text_nlp']} for rid in sorted({p['review_id'] for p in chunk})]
        cache = build_cache(output_path.parent/'tokens_evidence_add.jsonl', reviews, builder, cache=cache)
        rows = []
        for p in chunk:
            text = master[p['review_id']]['review_text_nlp']
            spans = []
            if p['predicted_label'] != 'NONE':
                c = cache[p['review_id']]
                windows = make_windows(c['ids'], ec['prefixes'][p['aspect']], [0]*len(c['ids']),
                                       ec['max_length'], ec['overlap'], tokenizer.cls_token_id, tokenizer.sep_token_id)
                sums, counts = np.zeros((len(c['ids']), 3)), np.zeros(len(c['ids']))
                for offset in range(0, len(windows), int(ec['config']['eval_batch_size'])):
                    group = windows[offset:offset+int(ec['config']['eval_batch_size'])]
                    batch = tokenizer.pad([{'input_ids': w['input_ids'], 'attention_mask': [1]*len(w['input_ids'])} for w in group], padding=True, return_tensors='pt')
                    with torch.inference_mode():
                        logits = model(**{k:v.to(target) for k,v in batch.items()}).logits.float().cpu().numpy()
                    for j,w in enumerate(group):
                        n = w['end']-w['start']
                        sums[w['start']:w['end']] += logits[j,w['prefix_length']:w['prefix_length']+n]
                        counts[w['start']:w['end']] += 1
                require(np.all(counts>0), 'Token bằng chứng chưa được bao phủ.')
                decoded = decode_bio(c['offsets'], (sums/counts[:,None]).argmax(-1).tolist(), text)
                spans = [{'start':s,'end':e,'text':text[s:e]} for s,e in decoded]
            status = ('not_mentioned_by_sentiment_model' if p['predicted_label']=='NONE' else
                      'needs_review_no_evidence' if not spans else
                      'mixed_evidence_polarity_not_separated' if p['predicted_label']=='MIX' else 'model_extracted_unverified')
            rows.append({**p,'evidence_text':' || '.join(s['text'] for s in spans),'evidence_spans_json':json.dumps(spans,ensure_ascii=False),
                         'evidence_status':status,'evidence_source':'model','evidence_model_version':profile['model_version'],
                         'evidence_text_sha256':text_hash(text)})
        write_csv(part, fields, rows)
        proof['parts'][str(start)] = sha(part)
        atomic_json(proof_path, proof)
        print(f'Đã trích bằng chứng {min(start+64,len(pairs))}/{len(pairs)} cặp.',flush=True)
    write_csv(output_path, fields, (r for path in paths for r in csv_rows(path)))
    require(evidence_bundle(cfg, profile) == profile, 'Bộ trích thay đổi trong lúc chạy.')
    proof.update(complete=True, output_sha256=sha(output_path))
    atomic_json(proof_path, proof)
    return output_path
