"""Biểu đồ, bảng tháng và bình luận của một cảnh báo xu hướng."""
from html import escape
import json
from math import ceil

import altair as alt
import pandas as pd
import streamlit as st

from dung_chung.bo_cuc import (
    chon, chon_binh_luan, huy_hieu, nhan_van_ban, thang_nam, the_so, van_ban_ngan,
)
from nhan_xet.chi_tiet import hien_bang_chung, hien_binh_luan_goc
from src.alerts.tang2.view_trend_alerts import comparison_months, pct


def hien_bieu_do(months, baseline_rate, tone='yellow'):
    data = pd.DataFrame(months, columns=['month', 'review_count', 'negative_count', 'complaint_rate'])
    data = data.astype({'review_count': int, 'negative_count': int, 'complaint_rate': float})
    data = data.sort_values('month').rename(columns={'complaint_rate': 'Tỷ lệ'})
    data['Tháng'] = data.month.map(thang_nam)
    data['Tỷ lệ hiển thị'] = data['Tỷ lệ'].map(pct)
    data['Tháng cảnh báo'] = data.month.eq(data.month.max())
    cap = min(1, max(0.1, ceil((max(data['Tỷ lệ'].max(), float(baseline_rate)) + .04) * 10) / 10))

    base = alt.Chart(data).encode(
        x=alt.X('Tháng:O', sort=data['Tháng'].tolist(), title=None,
                axis=alt.Axis(labelAngle=0, labelPadding=7, ticks=False),
                scale=alt.Scale(padding=1)),
        y=alt.Y('Tỷ lệ:Q', title=None, scale=alt.Scale(domain=[0, cap], nice=False),
                axis=alt.Axis(format='.0%', tickCount=4)),
    )
    marker = '#a32e25' if tone == 'red' else '#8e510a'
    highlight = alt.condition(alt.datum['Tháng cảnh báo'], alt.value(marker), alt.value('#0b7069'))
    reference = alt.Chart(pd.DataFrame({'Mốc so sánh': [float(baseline_rate)]})).mark_rule(
        color='#ad6b20', strokeDash=[6, 4], strokeWidth=1.6,
    ).encode(y=alt.Y('Mốc so sánh:Q', title=None, scale=alt.Scale(domain=[0, cap], nice=False)))
    area = base.mark_area(color='#d7eeea', opacity=.7)
    line = base.mark_line(color='#0b7069', strokeWidth=2.6)
    points = base.mark_circle(size=75, opacity=1, stroke='white', strokeWidth=1.5).encode(
        color=highlight,
        tooltip=[alt.Tooltip('Tháng:O'), alt.Tooltip('Tỷ lệ:Q', format='.1%'),
                 alt.Tooltip('negative_count:Q', title='Số tiêu cực'),
                 alt.Tooltip('review_count:Q', title='Tổng nhận xét')],
    )
    labels = base.mark_text(dy=-12, fontSize=13, fontWeight=700).encode(
        text='Tỷ lệ hiển thị:N', color=highlight,
    )
    chart = (area + reference + line + points + labels).properties(height=195).configure_axis(
        gridColor='#dce7ee', labelColor='#435972', labelFontSize=12, domain=False,
    ).configure_view(stroke=None)
    st.altair_chart(chart, use_container_width=True)
    st.markdown('<div class="chu-giai-xu-huong"><span>Tỷ lệ tiêu cực</span>'
                '<span>Trung bình các tháng dùng để so sánh</span></div>', unsafe_allow_html=True)


def hien_chi_tiet(ctx, row, mentions, key):
    months = comparison_months(row)
    with st.container(border=True, key='tang2_xu_huong'):
        st.subheader('Diễn biến phản ánh tiêu cực')
        st.markdown(
            f'<div class="dau-the"><strong>{escape(row["aspect_name"])}</strong>'
            f'{huy_hieu(row["level_title"], row["level"])}</div>', unsafe_allow_html=True,
        )
        st.caption(f"Tháng {thang_nam(row['month'])} so với {row['baseline_months']} tháng liền trước.")
        left, middle, right = st.columns(3)
        with left:
            the_so('Tỷ lệ tháng cảnh báo', pct(row['complaint_rate']),
                   f"{row['count_NEG']}/{row['review_count']} nhận xét", 'teal')
        with middle:
            the_so(f"Trung bình {row['baseline_months']} tháng trước", pct(row['baseline_rate']))
        with right:
            the_so('Mức tăng so với trung bình', f"{float(row['increase_pp']):+.1f}",
                   'điểm phần trăm', row['level'])
        st.markdown('#### Tỷ lệ tiêu cực theo tháng')
        hien_bieu_do(months, row['baseline_rate'], row['level'])

    with st.container(border=True, key='tang2_so_lieu_thang'):
        st.subheader('Số liệu từng tháng')
        rows = []
        alert_class = 'thang-canh-bao thang-canh-bao-do' if row['level'] == 'red' else 'thang-canh-bao'
        for month in months:
            current = f' class="{alert_class}"' if month['month'] == row['month'] else ''
            count = f"{month['negative_count']} / {month['review_count']}"
            rows.append(f"<tr{current}><td>{escape(thang_nam(month['month']))}</td>"
                        f"<td>{escape(count)}</td><td>{pct(month['complaint_rate'])}</td></tr>")
        st.markdown(
            '<table class="bang-xu-huong"><thead><tr><th scope="col">Tháng</th>'
            '<th scope="col">Tiêu cực / Tổng</th><th scope="col">Tỷ lệ</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>', unsafe_allow_html=True,
        )
        st.caption('Tỷ lệ tính trên tổng nhận xét của nhà hàng trong từng tháng.')

    with st.container(border=True, key='tang2_binh_luan_lien_quan'):
        hien_nhan_xet(ctx, row, months, mentions, key)


def hien_nhan_xet(ctx, row, months, mentions, key):
    st.subheader('Bình luận liên quan')
    time_col, kind_col = st.columns(2)
    with time_col:
        options = [item['month'] for item in reversed(months)]
        month = chon(options, {m: thang_nam(m) for m in options}, 'Tháng của bình luận', key + '_thang')
    with kind_col:
        kind = st.selectbox('Loại bình luận', ['Chỉ tiêu cực', 'Mọi bình luận có đề cập'],
                            key=key + '_loai')
    reader = ctx['t2']
    relevant = mentions.loc[mentions.review_month.eq(month)].set_index('review_id')
    labels = relevant['effective_label'].to_dict()
    if kind == 'Chỉ tiêu cực':
        ids = (json.loads(row['negative_review_ids_json']) if month == row['month'] else
               [rid for rid, label in labels.items() if label == 'NEG'])
    else:
        ids = list(labels)
    if not ids:
        st.info('Không có bình luận thuộc nhóm đang xem trong tháng này.')
        return

    ids.sort(key=lambda rid: reader.master[rid]['review_date_iso'], reverse=True)
    st.caption(f'{len(ids):,} bình luận về {row["aspect_name"].lower()} trong tháng đang xem.')
    options = {}
    for rid in ids:
        review = reader.master[rid]
        text = review.get('review_text') or review['review_text_nlp']
        label = 'NEG' if kind == 'Chỉ tiêu cực' else labels[rid]
        color = {'NEG': 'red', 'MIX': 'orange', 'POS': 'green', 'NEU': 'blue'}[label]
        options[rid] = (f'**{nhan_van_ban(rid)}** · :{color}[**{label}**]  \n'
                        f'{nhan_van_ban(van_ban_ngan(text))}')
    left, right = st.columns([1, 1.6])
    with left:
        selected = chon_binh_luan(options, key + '_review', limit=3)
    with right:
        review = reader.master[selected]
        review = dict(review, review_text=review.get('review_text') or review['review_text_nlp'])
        hien_binh_luan_goc(review)
        evidence = reader.evidence.get((selected, row['aspect']), {})
        needs_review = relevant.at[selected, 'evidence_needs_review'] if selected in relevant.index else True
        hien_bang_chung(evidence.get('evidence_text'), needs_review)
