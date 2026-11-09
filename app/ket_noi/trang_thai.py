"""Nối dữ liệu đã kiểm tra cho giao diện; không huấn luyện hoặc ghi kết quả."""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.modeling.phobert_system_common import load_settings
from src.pipeline.prepare_system_data import load_system_inputs
from src.alerts.view_content_alerts import BoDocTang1, score_review, TEN_VAN_DE
from src.alerts.tang2.view_trend_alerts import BoDocTang2
from src.alerts.tang2.build_trend_alerts import output_root, current_directory, read_json
from src.alerts.build_content_alerts import ASPECT_NAMES

TRANG_THAI = 'Đọc kết quả System và cảnh báo đã xây dựng'
NHAN = {'NONE':'Không đề cập','POS':'Tích cực','NEU':'Trung tính','NEG':'Tiêu cực','MIX':'Hỗn hợp'}
def dung(value):
    return str(value).lower() in {'true','1'}

def dau_vet(root):
    """Thay dữ liệu, con trỏ, cấu hình hoặc mã thì bộ nhớ đệm hết hiệu lực."""
    root=Path(root)
    files=set()
    for folder in ['reports/system/foody_v01','reports/alerts/foody_v01','data/system/foody_v01',
                   'configs/system','configs/alerts','src/pipeline','src/alerts','app']:
        directory=root/folder
        files.add(directory)
        if directory.is_dir():
            files.update(p for p in directory.rglob('*') if p.is_file()
                         and p.suffix in {'.json','.csv','.md','.txt','.py'})
    for folder in ['phobert_foody_add','phobert_evidence_add']:
        files.add(root/'models'/folder/'training_complete.json')
    return tuple((str(p.relative_to(root)),p.stat().st_mtime_ns,p.stat().st_size)
                 if p.exists() else (str(p.relative_to(root)),0,0) for p in sorted(files))

# Chỉ giữ các cột mà giao diện và hai bộ đọc cảnh báo sử dụng.
COT_BANG_CHUNG = (
    'review_id', 'aspect', 'label', 'label_source', 'label_verified', 'model_label',
    'evidence_text', 'evidence_source', 'evidence_model_version',
)
COT_CANH_BAO = (
    'review_id', 'record_type', 'aspect', 'severity', 'evidence_text',
    'evidence_needs_review', 'review_status', 'incident_types',
)
COT_KHIA_CANH = (
    *COT_BANG_CHUNG, 'evidence_needs_review', 'manual_label',
)


@st.cache_resource(show_spinner=False, max_entries=1)
def _nap(root, token):
    root = Path(root)
    ctx = {'root': root, 'errors': {}, 'system': None, 't1': None, 't2': None,
           'master': pd.DataFrame(), 'aspects': pd.DataFrame(), 'a8': pd.DataFrame(),
           'content': pd.DataFrame(), 'trend': pd.DataFrame(), 'coverage': pd.DataFrame(),
           'problems': pd.DataFrame()}

    with st.spinner('Đang kiểm tra và đọc dữ liệu System…'):
        try:
            inputs = load_system_inputs(project_root=root)
            master = pd.DataFrame(inputs['master'].values())
            master['review_month'] = master['review_month'].astype(str)
            if 'review_text' not in master:
                master['review_text'] = master['review_text_nlp']
            else:
                texts = master['review_text'].fillna('')
                master['review_text'] = texts.where(texts.ne(''), master['review_text_nlp'])
            aspects = pd.DataFrame.from_records(
                (tuple(row[column] for column in COT_KHIA_CANH) for row in inputs['aspects']),
                columns=COT_KHIA_CANH,
            )
            ctx.update(system=inputs['manifest'], paths=inputs['paths'], master=master,
                       aspects=aspects, a8=pd.DataFrame(inputs['a8']))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            ctx['errors']['system'] = str(exc)
            return ctx

    with st.spinner('Đang kiểm tra kết quả cảnh báo tầng 1…'):
        try:
            labels = ({'review_id': rid, 'label': label}
                      for rid, label in zip(aspects['review_id'], aspects['label']))
            reader = BoDocTang1(project_root=root, system_inputs=dict(inputs, aspects=labels))
            content = pd.read_csv(reader.paths['data'], usecols=COT_CANH_BAO,
                                  dtype=str, keep_default_na=False)
            rows = []
            for rid, counts in reader.label_counts.items():
                incident = reader.a8.get(rid)
                level = score_review(counts)['level'] if not incident else None
                if incident or level:
                    rows.append({'review_id': rid, 'kind': 'a8' if incident else 'a1_a7',
                                 'problem_level': 'A8 — Sự cố nghiêm trọng' if incident else TEN_VAN_DE[level],
                                 'review_status': incident['review_status'] if incident else ''})
            problems = pd.DataFrame(rows, columns=['review_id', 'kind', 'problem_level', 'review_status'])
            problems = problems.merge(master, on='review_id', how='left', validate='one_to_one')
            ctx.update(t1=reader, content=content, problems=problems)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            ctx['errors']['t1'] = str(exc)

    with st.spinner('Đang kiểm tra kết quả cảnh báo tầng 2…'):
        try:
            cfg = load_settings(project_root=root)
            cfg['_system_snapshot'] = inputs['paths']['directory']
            directory = current_directory(output_root(cfg))
            manifest = read_json(directory / 'trend_manifest.json')
            evidence_rows = aspects.loc[aspects['evidence_text'].ne(''), list(COT_BANG_CHUNG)]
            evidence = (dict(zip(COT_BANG_CHUNG, values))
                        for values in evidence_rows.itertuples(index=False, name=None))
            reader = BoDocTang2(project_root=root, as_of=manifest['signature']['as_of'],
                               system_inputs=dict(inputs, aspects=evidence))
            coverage = pd.read_csv(reader.directory / 'trend_coverage.csv', dtype=str, keep_default_na=False,
                                   usecols=['restaurant_id', 'restaurant_name', 'month', 'baseline_months',
                                            'review_count', 'text_group_count', 'status', 'status_title', 'reason'])
            ctx.update(t2=reader, trend=pd.DataFrame(reader.danh_sach()), coverage=coverage)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            ctx['errors']['t2'] = str(exc)
    return ctx


def nap_du_lieu(root):
    return _nap(str(Path(root).resolve()),dau_vet(root))

def master_scope(ctx, filters):
    rows=ctx['master']
    if rows.empty:return rows.copy()
    selected=rows
    if filters.get('restaurant'):
        selected=selected[selected.restaurant_id.eq(filters['restaurant'])]
    if 'months' in filters:
        selected = selected[selected.review_month.isin(filters['months'])]
    if filters.get('start'):selected=selected[selected.review_month.ge(filters['start'])]
    if filters.get('end'):selected=selected[selected.review_month.le(filters['end'])]
    if filters.get('aspect'):
        ids=ctx['aspects'].loc[ctx['aspects'].aspect.eq(filters['aspect']) & ctx['aspects'].label.ne('NONE'),'review_id']
        selected=selected[selected.review_id.isin(ids)]
    return selected.copy()

def aspect_scope(ctx, filters):
    rows=ctx['aspects']
    if rows.empty:return rows.copy()
    rows=rows[rows.review_id.isin(master_scope(ctx,filters).review_id)]
    if filters.get('aspect'):rows=rows[rows.aspect.eq(filters['aspect'])]
    return rows.copy()

def problems_scope(ctx, filters):
    rows=ctx['problems']
    if rows.empty:return rows.copy()
    rows=rows[rows.review_id.isin(master_scope(ctx,filters).review_id)]
    if filters.get('aspect'):
        content=ctx['content']
        ids=content.loc[content.aspect.eq(filters['aspect']),'review_id']
        rows=rows[rows.review_id.isin(ids)|rows.kind.eq('a8')]
    return rows.copy()

def content_scope(ctx, filters, record_type=None):
    rows = ctx['content']
    if rows.empty:
        return rows.copy()
    reviews = master_scope(ctx, filters)
    selected = rows[rows.review_id.isin(reviews.review_id)]
    if record_type:
        selected = selected[selected.record_type.eq(record_type)]
    return selected.copy()


def monthly_scope(ctx, filters, coverage=False):
    rows=ctx['coverage'] if coverage else ctx['trend']
    if rows.empty:return rows.copy()
    if filters.get('restaurant'):rows=rows[rows.restaurant_id.eq(filters['restaurant'])]
    if filters.get('start'):rows=rows[rows.month.ge(filters['start'])]
    if filters.get('end'):rows=rows[rows.month.le(filters['end'])]
    if not coverage and filters.get('aspect'):rows=rows[rows.aspect.eq(filters['aspect'])]
    return rows.copy()

def restaurants(ctx, filters):
    master=master_scope(ctx,filters)
    if master.empty:return pd.DataFrame(columns=['restaurant_id','restaurant_name','review_count','tier1','a8','tier2'])
    result=master.groupby(['restaurant_id','restaurant_name']).size().rename('review_count').reset_index()
    problems=problems_scope(ctx,filters)
    trend=monthly_scope(ctx,filters)
    for name,rows in [('tier1',problems),('a8',problems[problems.kind.eq('a8')] if not problems.empty else problems)]:
        counts=rows.groupby('restaurant_id').review_id.nunique() if not rows.empty else pd.Series(dtype=int)
        result[name]=result.restaurant_id.map(counts).fillna(0).astype(int) if ctx['t1'] else None
    result['tier2']=result.restaurant_id.isin(trend.restaurant_id) if not trend.empty else False
    if not ctx['t2']:result['tier2']=None
    return result.sort_values(['review_count','restaurant_name'],ascending=[False,True])

def require_part(ctx, part):
    if part=='system' and ctx['system'] is not None:return True
    if part!='system' and ctx.get(part) is not None:return True
    title={'system':'System','t1':'cảnh báo tầng 1','t2':'cảnh báo tầng 2'}[part]
    st.info('Chưa có kết quả '+title+' hợp lệ để hiển thị.')
    error=ctx['errors'].get(part) or ctx['errors'].get('system')
    if error:
        with st.expander('Lý do chưa đọc được '+title):
            st.code(error,language=None)
    return False

