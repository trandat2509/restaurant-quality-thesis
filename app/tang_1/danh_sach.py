"""Cảnh báo khía cạnh và sự cố bất thường của nhà hàng đang chọn."""
import streamlit as st

from dung_chung.bo_cuc import chon, thang_nam, tieu_de
from ket_noi.trang_thai import ASPECT_NAMES, content_scope, master_scope, require_part
from tang_1.chi_tiet import hien_binh_luan, hien_su_co

KHIA_CANH = 'Theo khía cạnh'
SU_CO = 'Sự cố bất thường'


def hien_trang(ctx, filters):
    tieu_de('Cảnh báo tầng 1', 'Xem từng bình luận và các khía cạnh cần chú ý.')
    if not require_part(ctx, 't1'):
        return

    restaurant_id = filters['restaurant']
    reviews = master_scope(ctx, filters)
    st.caption(reviews.iloc[0]['restaurant_name'])
    section = st.radio('Mục cảnh báo tầng 1', [KHIA_CANH, SU_CO], horizontal=True,
                       label_visibility='collapsed', key='tang1_muc')

    scoped_restaurant, chain_months = st.session_state.get('tang1_chuoi') or (None, ())
    chain_selected = scoped_restaurant == restaurant_id
    months = sorted(reviews.review_month.unique(), reverse=True)
    options = (['chuoi'] if chain_selected else []) + ['', *months]
    labels = {'chuoi': 'Các tháng trong chuỗi xu hướng', '': 'Tất cả các tháng',
              **{month: thang_nam(month) for month in months}}
    left, right = st.columns(2)
    with left:
        month = chon(options, labels, 'Tháng/năm của bình luận', 'tang1_thang')
    if month == 'chuoi':
        filters = {**filters, 'months': chain_months}
        if chain_months:
            st.caption('Các tháng trong chuỗi · ' + ', '.join(map(thang_nam, chain_months)))
    elif month:
        filters = {**filters, 'months': (month,)}

    incidents = section == SU_CO
    if not incidents:
        with right:
            aspect = chon(['', *ASPECT_NAMES], {'': 'Tất cả khía cạnh', **ASPECT_NAMES},
                          'Khía cạnh cảnh báo', 'tang1_khia_canh')

    if month == 'chuoi' and not chain_months:
        st.info('Chưa có chuỗi xu hướng trong phạm vi đang xem.')
        return

    alerts = content_scope(ctx, filters, 'a8_incident' if incidents else 'aspect_reflection')
    if not incidents and aspect:
        alerts = alerts.loc[alerts.aspect.eq(aspect)]
    if alerts.empty:
        st.info('Không có sự cố bất thường xảy ra.' if incidents else
                'Không có bình luận cảnh báo cho khía cạnh và tháng đang xem.')
        return

    matches = reviews.loc[reviews.review_id.isin(alerts.review_id)].sort_values(
        ['review_date_iso', 'review_id'], ascending=[False, True])
    options = {
        row.review_id: f"{index}. {thang_nam(row.review_month)} · {' '.join(row.review_text.split())[:90]}"
        for index, row in enumerate(matches.itertuples(), 1)
    }
    st.caption(f'{len(matches)} bình luận trong phạm vi đang xem')
    key = 'tang1_binh_luan_su_co' if incidents else 'tang1_binh_luan_khia_canh'
    review_id = chon(options, options, 'Bình luận', key)
    review = matches.set_index('review_id').loc[review_id]
    selected = ctx['content'].loc[ctx['content'].review_id.eq(review_id)]
    if incidents:
        incident = selected.loc[selected.record_type.eq('a8_incident')].iloc[0]
        hien_su_co(review, incident)
    else:
        levels = selected.loc[selected.record_type.eq('aspect_reflection')].set_index('aspect').severity
        problem = ctx['problems'].loc[ctx['problems'].review_id.eq(review_id)].iloc[0]
        aspects = ctx['aspects']
        aspects = aspects.loc[aspects.review_id.eq(review_id) & aspects.label.ne('NONE')]
        hien_binh_luan(review, problem, aspects, levels)
