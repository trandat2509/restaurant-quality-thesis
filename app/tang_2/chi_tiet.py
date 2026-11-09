"""Chuỗi tháng, tỷ lệ và nhận xét liên quan đến cảnh báo."""
import json

import pandas as pd
import streamlit as st

from dung_chung.bo_cuc import chi_so, chon, huy_hieu
from nhan_xet.chi_tiet import hien_bang_chung
from src.alerts.tang2.view_trend_alerts import alert_description, comparison_months, pct, render_chart


def hien_bieu_do(months, baseline_rate):
    chart = render_chart(months, reference_rate=baseline_rate)
    st.markdown('<div class="bieu-do-xu-huong">' + chart + '</div>', unsafe_allow_html=True)


def hien_chi_tiet(ctx, row, mentions, key):
    st.markdown(huy_hieu(row['level_title'], row['level']), unsafe_allow_html=True)
    st.write(alert_description(row))
    chi_so({
        'Tỷ lệ tháng cảnh báo': pct(row['complaint_rate']),
        f"Trung bình {row['baseline_months']} tháng trước": pct(row['baseline_rate']),
        'Mức tăng': f"+{float(row['increase_pp']):.1f} điểm %",
    })
    months = comparison_months(row)
    hien_bieu_do(months, row['baseline_rate'])
    table = pd.DataFrame([
        {'Tháng': month['month'],
         'Tiêu cực / tổng nhận xét': f"{month['negative_count']}/{month['review_count']}",
         'Tỷ lệ tiêu cực': pct(month['complaint_rate'])}
        for month in months
    ])
    st.dataframe(table, hide_index=True, use_container_width=True)
    hien_nhan_xet(ctx, row, months, mentions, key)


def hien_nhan_xet(ctx, row, months, mentions, key):
    st.markdown('##### Nhận xét về khía cạnh này')
    month = st.selectbox('Tháng của nhận xét', [item['month'] for item in reversed(months)],
                         key=key + '_thang')
    kind = st.selectbox('Loại nhận xét', ['Chỉ tiêu cực', 'Mọi nhận xét có đề cập'],
                        key=key + '_loai')
    reader = ctx['t2']
    labels = mentions.loc[mentions.review_month.eq(month)].set_index('review_id')['effective_label'].to_dict()
    if kind == 'Chỉ tiêu cực' and month == row['month']:
        ids = json.loads(row['negative_review_ids_json'])
    elif kind == 'Chỉ tiêu cực':
        ids = [rid for rid, label in labels.items() if label == 'NEG']
    else:
        ids = list(labels)
    if not ids:
        st.info('Không có nhận xét thuộc nhóm đang xem trong tháng này.')
        return
    ids.sort(key=lambda rid: reader.master[rid]['review_date_iso'], reverse=True)
    options = {rid: f"{reader.master[rid]['review_date_iso'][:10]} · {rid}" for rid in ids}
    st.caption(f'{len(ids)} nhận xét trong nhóm đang xem')
    selected = chon(ids, options, 'Đọc nhận xét liên quan', key + '_review')
    review = reader.master[selected]
    label = 'NEG' if kind == 'Chỉ tiêu cực' else labels[selected]
    st.markdown(huy_hieu(label, label), unsafe_allow_html=True)
    st.text(review.get('review_text') or review['review_text_nlp'])
    evidence = reader.evidence.get((selected, row['aspect']))
    if evidence:
        hien_bang_chung(evidence['evidence_text'])
