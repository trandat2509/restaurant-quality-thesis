"""Cảnh báo xu hướng theo khía cạnh của nhà hàng đang chọn."""
import streamlit as st

from dung_chung.bo_cuc import chon, nhan_van_ban, thang_nam
from ket_noi.trang_thai import dung, master_scope, monthly_scope, require_part
from src.alerts.tang2.view_trend_alerts import empty_message, pct
from tang_2.chi_tiet import hien_chi_tiet


def hien_trang(ctx, filters):
    if not require_part(ctx, 't2'):
        return

    restaurant_id = filters['restaurant']
    rows = monthly_scope(ctx, filters)
    if rows.empty:
        coverage = monthly_scope(ctx, filters, coverage=True)
        st.info(empty_message(coverage.to_dict('records')))
        return

    grouped = dict(list(rows.groupby('aspect', sort=False)))
    labels = {}
    for aspect, events in grouped.items():
        event = events.iloc[0]
        tone = 'red' if event['level'] == 'red' else 'orange'
        labels[aspect] = (
            f"**{nhan_van_ban(event['aspect_name'])}**  \n"
            f":{tone}[**{nhan_van_ban(event['level_title'])}**] · "
            f"**{pct(event['complaint_rate'])}** tiêu cực · **{float(event['increase_pp']):+.1f} điểm phần trăm**"
        )

    target = st.session_state.pop('tang2_muc_tieu', None)
    aspect_key = f'tang2_khia_canh_{restaurant_id}'
    if target and target[0] == restaurant_id and target[1] in grouped:
        st.session_state[aspect_key] = target[1]
    if st.session_state.get(aspect_key) not in grouped:
        st.session_state[aspect_key] = next(iter(grouped))

    st.subheader('Khía cạnh có cảnh báo')
    aspect = st.radio('Chọn khía cạnh có cảnh báo', list(grouped), horizontal=True,
                      format_func=labels.__getitem__, key=aspect_key, label_visibility='collapsed')
    events = grouped[aspect]
    event_key = f'tang2_dot_{restaurant_id}_{aspect}'
    options = list(zip(events.month, events.baseline_months.astype(int)))
    if target and target[:2] == (restaurant_id, aspect) and target[2:] in options:
        st.session_state[event_key] = target[2:]
    selected = next(iter(options))
    if len(options) > 1:
        event_labels = {key: f'{thang_nam(key[0])} · So với {key[1]} tháng trước'
                        for key in options}
        selected = chon(options, event_labels, 'Lần phát cảnh báo', event_key)

    month, window = selected
    row = events.loc[events.month.eq(month) & events.baseline_months.astype(int).eq(window)].iloc[0]
    reviews = master_scope(ctx, {'restaurant': restaurant_id})
    aspects = ctx['aspects']
    mentions = aspects.loc[aspects.review_id.isin(reviews.review_id) & aspects.aspect.eq(aspect)].copy()
    confirmed = mentions.label_source.eq('manual') & mentions.label_verified.map(dung)
    manual = mentions.manual_label.where(mentions.manual_label.ne(''), mentions.label)
    mentions['effective_label'] = mentions.model_label.where(~confirmed, manual)
    mentions = mentions[mentions.effective_label.isin(['POS', 'NEU', 'NEG', 'MIX'])].copy()
    mentions['review_month'] = mentions.review_id.map(reviews.set_index('review_id').review_month)

    key = f'tang2_{restaurant_id}_{aspect}_{month}_{window}'
    hien_chi_tiet(ctx, row, mentions, key)
