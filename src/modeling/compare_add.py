"""So sánh các mô hình trên đúng cùng nhãn chuẩn và cùng phần kiểm tra."""
import json
import pandas as pd
from .common_add import ASPECTS,LABELS
from .data_add import read_csv
from .evidence_core_add import read_jsonl,evaluate_spans
from src.training.prepare_foody_absa_add import save_json


def run(root,cfg):
    base=root/cfg['report_dir'];names=['tfidf','phobert_base','phobert_vlsp'];frames={};rows=[]
    for name in names:
        report=base/(name+'_add');summary=json.loads((report/'metrics_summary_add.json').read_text(encoding='utf-8'))
        frames[name]=read_csv(report/'out_of_fold_predictions_add.csv')
        rows.append({'model':name+'_add',**{k:summary[k] for k in ['review_count','macro_f1_7_aspects_5_labels','fold_macro_f1_mean','fold_macro_f1_std','exact_review_accuracy']}})
    reference=frames[names[0]]
    key=['review_id','aspect','gold','fold','text_group_id']
    for n,d in frames.items():
        if d[key].sort_values(key).reset_index(drop=True).equals(reference[key].sort_values(key).reset_index(drop=True)) is False:
            raise ValueError(f'{n} không dùng cùng mẫu và cùng phần kiểm tra.')
    out=base/'comparison_add';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'comparison_add.csv',index=False,encoding='utf-8-sig')
    # Điểm trích bằng chứng khi biết nhãn thật và khi nhánh nhãn thật sự được dùng.
    evidence=read_jsonl(base/'evidence_add/out_of_fold_predictions_add.jsonl')
    lookup={(r.review_id,r.aspect):r.pred for r in frames[cfg['final_classifier']].itertuples()}
    gated=[]
    for r in evidence:
        if lookup[(r['review_id'],r['aspect'])]=='NONE':r={**r,'predicted_spans':[]}
        gated.append(r)
    save_json(out/'evidence_pipeline_metrics_add.json',{'with_gold_aspect_labels':evaluate_spans(evidence),'with_predicted_aspect_labels':evaluate_spans(gated),
        'note':'Bằng chứng không phân loại cực tính riêng từng đoạn. Nhãn MIX không bảo đảm hai đoạn tương ứng POS/NEG.',
        'ambiguous_gold_position_pairs_excluded':True,'independent_final_test_claimed':False})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
