"""Nhập kết quả đầy đủ đã dự đoán và tạo cả hai tầng cảnh báo.

Chạy từ gốc dự án: python -m src.pipeline.cap_nhat_foody_add --apply --device cuda
Không chạy lại phân loại hoặc trích bằng chứng; không thay tệp nguồn.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

from src.modeling.phobert_system_common import ASPECTS, LABELS, load_settings, location
from src.pipeline.prepare_system_data import (
    csv_rows, jread, require, sha, text_hash, atomic_json, model_evidence, validate_probability,
    manual_evidence, source_evidence, SOURCE_EVIDENCE_COLUMNS,
)
from src.pipeline.update_system_data import safe_id, write_csv


def prepare(cfg, batch_id='foody_day_du', prediction_file=None, manual_file=None):
    """Chỉ chuẩn bị đầu vào và chứng từ; chưa đăng ký trọng số hoặc kích hoạt system."""
    root = cfg['_root']
    prediction_file = Path(prediction_file) if prediction_file else root/'data/predictions/foody_add/aspect_predictions_with_evidence_add.csv'
    manual_file = Path(manual_file) if manual_file else root/'data/training/foody_add/absa_all_add.csv'
    prediction_file = prediction_file.resolve()
    manual_file = manual_file.resolve()
    base = location(cfg, 'system_data_dir')
    master_path = base/'reviews_master.csv'
    proof_path = prediction_file.parent/'prediction_complete_add.json'
    proof = jread(proof_path)
    require(proof.get('identity',{}).get('format',0)>=3 and proof.get('separate_model_and_manual_evidence') is True,
            'Cần bảng mới có nhãn/bằng chứng máy và tay ở các cột riêng. Không dùng bảng gộp cũ.')
    require(proof.get('reviews') == cfg['expected_corpus_rows'] and proof.get('rows') == cfg['expected_corpus_rows']*7,
            'Đầu ra nguồn chưa phủ toàn corpus.')
    require(sha(prediction_file) == proof['sha256'], 'Bảng kết quả khác hồ sơ hoàn tất.')
    require(proof['output_sha256'].get(prediction_file.name) == proof['sha256'], 'Hồ sơ hai bảng kết quả không thống nhất.')
    require(sha(master_path) == proof['identity']['corpus_sha256'], 'Bảng kết quả dùng corpus khác.')
    master = {r['review_id']:r for r in csv_rows(master_path)}
    require(len(master) == proof['reviews'], 'Sai số nhận xét corpus.')
    raw_manual = list(csv_rows(manual_file))
    require(sha(manual_file) == proof['identity']['manual_sha256'], 'Bộ nhãn tay khác nguồn đã ghép khi dự đoán.')
    manual = {r['review_id']:r for r in raw_manual}
    require(len(manual) == len(raw_manual) and manual, 'Nhãn tay trống hoặc trùng mã.')
    for rid,r in manual.items():
        require(rid in master and r['review_text_nlp'] == master[rid]['review_text_nlp']
                and r['text_group_id'] == master[rid]['text_group_id'], 'Nhãn tay khác corpus: '+rid)
        require(str(r.get('uncertain','0')) in {'0','false','False',''}, 'Chưa xác nhận nhận xét: '+rid)
        require(all(r[a] in LABELS for a in ASPECTS), 'Nhãn tay ngoài bộ nhãn.')
    machine_path=prediction_file.parent/'aspect_predictions_model_add.csv'
    machine_sha=proof['output_sha256'].get(machine_path.name)
    require(machine_sha, 'Hồ sơ thiếu mã băm của bảng máy.')
    if machine_path.is_file():require(sha(machine_path)==machine_sha, 'Bảng máy khác hồ sơ hoàn tất.')
    machine_rows=iter(csv_rows(machine_path)) if machine_path.is_file() else None
    model_profiles = {
        'model_profile.json': {'model_version':'phobert_foody_add','model_dir':'models/phobert_foody_add'},
        'evidence_profile.json': {'format':'foody_add','model_version':'phobert_evidence_add','model_dir':'models/phobert_evidence_add'}}
    for profile in model_profiles.values():
        md = root/profile['model_dir']
        require(jread(md/'training_complete.json').get('training_completed') is True, 'Mô hình chưa hoàn tất.')
        require(sha(md/'training_complete.json') == proof['identity']['model_signatures'][profile['model_version']],
                'Mô hình không thuộc lần dự đoán nguồn.')
        profile['vncorenlp_dir'] = cfg['vncorenlp_dir']
    identity = {'prediction_sha256':sha(prediction_file),'manual_sha256':sha(manual_file),
                'corpus_sha256':sha(master_path),'model_profiles':model_profiles,'format':2,
                'prediction_proof_sha256':sha(proof_path),'machine_sha256':machine_sha}
    final = root/'data/incoming'/safe_id(batch_id)
    if final.exists():
        old = jread(final/'import_proof.json')
        require(old['identity'] == identity, 'Mã đợt đã có nguồn khác; dùng --batch-id mới.')
        require(all((final/n).is_file() and sha(final/n)==h for n,h in old['output_sha256'].items()),
                'Đầu vào đã chuẩn bị bị thay đổi; không ghi đè lần đang tiếp tục.')
        return final/'batch.json', old['summary']
    final.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='dang_chuan_bi_',dir=final.parent))
    try:
        machine_fields = ['review_id','aspect','predicted_label','label_source','model_version','confidence','margin',
                          *['p_'+x for x in LABELS],'review_text_sha256']
        evidence_fields = ['review_id','aspect','predicted_label','model_version','evidence_text','evidence_source',
                           'evidence_model_version','evidence_spans_json','evidence_status','evidence_text_sha256']
        counts = {'manual_reviews':len(manual),'machine_reviews_existing':0,'machine_pairs_existing':0,
                  'manual_position_ambiguous_pairs':0,'manual_multiple_fragment_cells':0,
                  'machine_reviews_to_predict':0,'evidence_pairs_to_predict':0}
        seen, machine_ids = set(),set()
        with (staging/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as mf, (staging/'evidence.csv').open('w',encoding='utf-8-sig',newline='') as ef:
            mw,ew = csv.DictWriter(mf,fieldnames=machine_fields),csv.DictWriter(ef,fieldnames=evidence_fields)
            mw.writeheader();ew.writeheader()
            for r in csv_rows(prediction_file):
                rid,a = r['review_id'],r['aspect']
                require(rid in master and a in ASPECTS and (rid,a) not in seen, 'Trùng cặp hoặc dự đoán ngoài corpus.')
                seen.add((rid,a))
                text=master[rid]['review_text_nlp'];digest=text_hash(text)
                require(r['model_version']=='phobert_foody_add', 'Sai phiên mô hình cảm xúc nguồn.')
                require(r.get('review_text_sha256')==digest, 'Mã băm nhận xét khác corpus: '+rid)
                normalized = {k:r.get(k,'') for k in machine_fields}
                normalized.update(predicted_label=r['model_label'],label_source='model',
                                  confidence=r['model_confidence'],margin=r['model_margin'])
                require(all(str(normalized.get('p_'+x,'')).strip() for x in LABELS), 'Đầu ra mới thiếu xác suất đầy đủ.')
                validate_probability(normalized)
                er = {k:r.get(k,'') for k in evidence_fields}
                raw_model=source_evidence(r,'model')
                er.update({k:v for k,v in raw_model.items() if k in evidence_fields})
                er['predicted_label']=normalized['predicted_label']
                require(er['evidence_source']=='model' and er['evidence_model_version']=='phobert_evidence_add', 'Sai nguồn/phiên bằng chứng máy.')
                model_evidence(text,er)
                if rid in manual:
                    gold=manual[rid];hand=source_evidence(r,'manual')
                    require(r['manual_label']==gold[a] and hand['evidence_text']==gold[a+'_evidence']
                            and hand['evidence_source']=='manual' and not hand['evidence_model_version']
                            and hand['evidence_text_sha256']==digest,'Nhãn/bằng chứng tay độc lập không khớp nguồn: '+rid)
                    checked=manual_evidence(text,gold[a],gold[a+'_evidence'])
                    require(json.loads(hand['evidence_spans_json'])==json.loads(checked['evidence_spans_json']), 'Vị trí bằng chứng tay khác nguồn: '+rid)
                    require(r['label_source']=='manual' and r['label_verified']=='true' and r['predicted_label']==gold[a]
                            and r['evidence_text']==gold[a+'_evidence'] and r['evidence_source']=='manual', 'Cột ưu tiên không giữ nhãn/bằng chứng tay.')
                    counts['manual_position_ambiguous_pairs'] += 'ambiguous_manual_position' in json.loads(checked['evidence_quality_flags_json'])
                else:
                    require(not r.get('manual_label') and all(not r.get(col) for col in SOURCE_EVIDENCE_COLUMNS['manual'].values()), 'Có nhãn/bằng chứng tay ngoài bộ gán tay.')
                    require(r['label_source']=='model' and r['label_verified']=='false' and r['predicted_label']==normalized['predicted_label']
                            and all(r.get(k)==er[k] for k in ['evidence_text','evidence_source','evidence_model_version','evidence_spans_json','evidence_status','evidence_text_sha256']),
                            'Cột ưu tiên khác nguồn máy.')
                if machine_rows is not None:
                    other=next(machine_rows,None)
                    require(other is not None and all(other.get(k)==v for k,v in normalized.items())
                            and all(other.get(k)==v for k,v in er.items()),'Hai bảng máy và tổng hợp không khớp từng cặp.')
                mw.writerow(normalized)
                ew.writerow(er)
                machine_ids.add(rid);counts['machine_pairs_existing']+=1
        require(len(seen)==len(master)*7, 'Bảng nguồn chưa đủ bảy khía cạnh mỗi nhận xét.')
        require(machine_rows is None or next(machine_rows,None) is None, 'Bảng máy còn dòng thừa.')
        counts['machine_reviews_existing']=len(machine_ids)
        require(counts['machine_pairs_existing']==len(machine_ids)*7, 'Phần máy thiếu khía cạnh.')
        require(machine_ids==set(master), 'Chưa có nhãn máy đầy đủ cho toàn bộ corpus; không tự chạy lại mô hình.')
        counts['manual_multiple_fragment_cells']=sum(len([s for s in r[a+'_evidence'].split('||') if s.strip()])>2 for r in raw_manual for a in ASPECTS)
        write_csv(staging/'manual.csv',list(raw_manual[0]),raw_manual)
        for name,profile in model_profiles.items():atomic_json(staging/name,profile)
        batch = {'batch_id':batch_id,'annotation_version':'foody_add_confirmed','manual_confirmed':True,
                 'replace_manual':True,'replace_predictions':True,'manual_file':'manual.csv',
                 'predictions_file':'predictions.csv','evidence_file':'evidence.csv',
                 'source_manifest_file':'import_proof.json',
                 'model_profile_file':'model_profile.json','evidence_profile_file':'evidence_profile.json',
                 'note':'Nhập đủ nhãn/bằng chứng máy đã xuất, giữ nguồn tay riêng. Không chạy lại dự đoán; kế thừa A8 và lịch sử huấn luyện.'}
        atomic_json(staging/'batch.json',batch)
        output_names=['manual.csv','predictions.csv','evidence.csv',*model_profiles,'batch.json']
        training_config_path = root/'configs/training_add/training_add.json'
        excluded_old = jread(training_config_path).get('excluded_unconfirmed_add',[]) if training_config_path.is_file() else []
        annotation_audit = {'old_config_exclusions_present_in_confirmed_input':
                            [r['review_id'] for r in excluded_old if r['review_id'] in manual],
                            'operational_decision':'Giữ nhãn đã xác nhận trong manual.csv; không sửa trọng số hoặc hồ sơ huấn luyện gốc.'}
        atomic_json(staging/'import_proof.json',{'identity':identity,'summary':counts,'annotation_audit':annotation_audit,
                    'output_sha256':{n:sha(staging/n) for n in output_names}})
        require(sha(prediction_file)==identity['prediction_sha256'] and sha(manual_file)==identity['manual_sha256']
                and sha(master_path)==identity['corpus_sha256'] and sha(proof_path)==identity['prediction_proof_sha256'],
                'Nguồn thay đổi trong lúc chuẩn bị; chưa nhập vào hệ thống.')
        staging.rename(final)
        return final/'batch.json',counts
    finally:
        if staging.exists():shutil.rmtree(staging)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=None)
    p.add_argument('--batch-id',default='foody_day_du')
    p.add_argument('--predictions',type=Path,default=None)
    p.add_argument('--manual',type=Path,default=None)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--system-only',action='store_true',help='Chỉ nhập/cập nhật System; chạy cảnh báo riêng sau.')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args(argv)
    require(not args.system_only or args.apply, '--system-only cần dùng cùng --apply.')
    cfg=load_settings(project_root=args.project_root)
    batch,summary=prepare(cfg,args.batch_id,args.predictions,args.manual)
    print('Đầu vào đã chuẩn bị:',batch)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if args.apply:
        from src.pipeline.update_system_data import apply_batch, console_result
        result=apply_batch(cfg,batch,run_predictions=False,run_evidence=False,device=args.device,
                           build_alerts=not args.system_only)
        print(json.dumps(console_result(result),ensure_ascii=False,indent=2))
    else:
        print('Chưa cập nhật system. Thêm --apply để nhập kết quả đã có và tạo hai tầng cảnh báo; không chạy lại mô hình.')
    return 0


if __name__=='__main__':main()
