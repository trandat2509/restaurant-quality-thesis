"""Hiển thị tất cả khía cạnh có cảnh báo của nhà hàng đang chọn."""
import streamlit as st

from dung_chung.bo_cuc import tieu_de
from ket_noi.trang_thai import dung, master_scope, monthly_scope, require_part
from src.alerts.tang2.view_trend_alerts import empty_message
from tang_2.chi_tiet import hien_chi_tiet


def hien_trang(ctx, filters):
    tieu_de('Cảnh báo tầng 2', 'Theo dõi phản ánh tiêu cực theo từng khía cạnh và từng tháng.')
    if not require_part(ctx, 't2'):
        return

    restaurant_id = filters['restaurant']
    reviews = master_scope(ctx, {'restaurant': restaurant_id})
    st.caption(reviews.iloc[0]['restaurant_name'])
    rows = monthly_scope(ctx, filters)
    if rows.empty:
        coverage = monthly_scope(ctx, filters, coverage=True)
        st.info(empty_message(coverage.to_dict('records')))
        return

    # Giữ cả các tháng làm mốc khi phạm vi đang lọc một tháng phát cảnh báo.
    aspects = ctx['aspects']
    mentions = aspects.loc[aspects.review_id.isin(reviews.review_id) & aspects.aspect.isin(rows.aspect)].copy()
    confirmed = mentions.label_source.eq('manual') & mentions.label_verified.map(dung)
    manual = mentions.manual_label.where(mentions.manual_label.ne(''), mentions.label)
    mentions['effective_label'] = mentions.model_label.where(~confirmed, manual)
    mentions = mentions[mentions.effective_label.isin(['POS', 'NEU', 'NEG', 'MIX'])]
    mentions['review_month'] = mentions.review_id.map(reviews.set_index('review_id').review_month)

    latest = rows.index[0]
    for aspect, events in rows.groupby('aspect', sort=True):
        st.markdown('#### ' + events.iloc[0]['aspect_name'])
        related = mentions[mentions.aspect.eq(aspect)]
        for index, row in events.iterrows():
            month, window = row['month'], row['baseline_months']
            title = f"{month[5:]}/{month[:4]} · {row['level_title']} · So với {window} tháng trước"
            event_key = f'tang2_{restaurant_id}_{aspect}_{month}_{window}'
            with st.expander(title, expanded=index == latest):
                hien_chi_tiet(ctx, row, related, key=event_key)
