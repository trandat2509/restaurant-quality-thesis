"""Thẻ bình luận, khía cạnh và sự cố dùng chung trên giao diện."""
from html import escape

import streamlit as st

from dung_chung.bo_cuc import huy_hieu, thang_nam
from ket_noi.trang_thai import ASPECT_NAMES, dung
from src.alerts.view_content_alerts import TEN_MUC_DO, ten_su_co

MAU_MUC_DO = {'Nhẹ': 'nhe', 'Vừa': 'vua', 'Nặng': 'nang'}
MAU_KHIA_CANH = {'light': 'vua', 'serious': 'nang', 'unknown': 'NONE'}


def hien_bang_chung(evidence, needs_review=False):
    if evidence and not dung(needs_review):
        st.caption('Bằng chứng')
        text = escape(evidence.replace(' || ', '\n')).replace('\n', '<br>')
        st.markdown(f'<blockquote class="doan-bang-chung">{text}</blockquote>',
                    unsafe_allow_html=True)
    else:
        st.caption('Cần xem xét bình luận gốc.')


def hien_binh_luan_goc(review, muc_do=None):
    with st.container(border=True, key='the_binh_luan_goc'):
        badge = huy_hieu(f'Mức độ cảnh báo · {muc_do}', MAU_MUC_DO[muc_do]) if muc_do else ''
        month = escape(thang_nam(review['review_month']))
        st.markdown(f'<div class="dau-the"><h4>Bình luận gốc</h4>'
                    f'<span class="thang-binh-luan">{month}</span>{badge}</div>',
                    unsafe_allow_html=True)
        st.text(review['review_text'])


def hien_khia_canh(aspects, levels=None):
    rows = list(aspects.sort_values('aspect').itertuples())
    with st.container(key='luoi_khia_canh'):
        for start in range(0, len(rows), 2):
            pair = rows[start:start + 2]
            for col, row in zip(st.columns(len(pair)), pair):
                with col.container(border=True, key=f'the_khia_canh_{row.aspect}'):
                    level = levels.get(row.aspect) if levels is not None else None
                    severity = huy_hieu(TEN_MUC_DO[level], MAU_KHIA_CANH[level]) if level else ''
                    st.markdown(f'<div class="dau-the"><strong>{escape(ASPECT_NAMES[row.aspect])}</strong>'
                                f'{huy_hieu(row.label, row.label)}{severity}</div>', unsafe_allow_html=True)
                    if row.label != 'NONE':
                        hien_bang_chung(row.evidence_text, row.evidence_needs_review)


def hien_su_co(review, incident_types, status, evidence, needs_review):
    confirmed = status == 'DA_DUYET'
    state = huy_hieu('Đã xác nhận' if confirmed else 'Cần xem xét', 'POS' if confirmed else 'vua')
    with st.container(border=True, key='the_su_co'):
        month = escape(thang_nam(review['review_month']))
        st.markdown(f'<div class="dau-the"><h4>Sự cố bất thường</h4>{state}'
                    f'<span class="thang-binh-luan">{month}</span></div>', unsafe_allow_html=True)
        st.text(' · '.join(ten_su_co(incident_types)))
        st.markdown('**Bình luận gốc**')
        st.text(review['review_text'])
        hien_bang_chung(evidence, needs_review)


def hien_chi_tiet(ctx, review_id=None, key='nhan_xet'):
    if not review_id:
        return
    match = ctx['master'].loc[ctx['master'].review_id.eq(review_id)]
    if match.empty:
        st.info('Không tìm thấy nhận xét.')
        return
    review = match.iloc[0]
    st.markdown('#### Chi tiết nhận xét')
    st.caption(f"{review_id} · {review['restaurant_name']}")

    a8 = ctx['a8']
    incidents = a8.loc[a8.review_id.eq(review_id)] if not a8.empty else a8
    if incidents.empty:
        hien_binh_luan_goc(review)
    else:
        incident = incidents.iloc[0]
        hien_su_co(review, incident.incident_types, incident.review_status, incident.A8_evidence,
                   incident.review_status != 'DA_DUYET')

    st.markdown('##### Phân tích theo khía cạnh')
    aspects = ctx['aspects']
    hien_khia_canh(aspects.loc[aspects.review_id.eq(review_id)])
