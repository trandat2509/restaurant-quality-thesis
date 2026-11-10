"""Danh sách chung cảnh báo khía cạnh và sự cố bất thường."""
import streamlit as st

from dung_chung.bo_cuc import chon, chon_binh_luan, nhan_van_ban, thang_nam, van_ban_ngan
from ket_noi.trang_thai import ASPECT_NAMES, content_scope, master_scope, require_part
from tang_1.chi_tiet import hien_binh_luan


def hien_trang(ctx, filters, controls):
    if not require_part(ctx, 't1'):
        return

    restaurant_id = filters['restaurant']
    reviews = master_scope(ctx, filters)
    if reviews.empty:
        st.info('Chưa có nhận xét của nhà hàng này.')
        return

    scoped_restaurant, chain_months = st.session_state.get('tang1_chuoi') or (None, ())
    chain_selected = scoped_restaurant == restaurant_id and bool(chain_months)
    months = sorted(reviews.review_month.unique(), reverse=True)
    options = (['chuoi'] if chain_selected else []) + ['', *months]
    labels = {'chuoi': 'Các tháng trong chuỗi xu hướng', '': 'Tất cả các tháng',
              **{month: thang_nam(month) for month in months}}
    month_col, aspect_col = controls
    with month_col:
        month = chon(options, labels, 'Tháng/năm của bình luận', 'tang1_thang')
    with aspect_col:
        aspect = chon(['', *ASPECT_NAMES], {'': 'Tất cả khía cạnh', **ASPECT_NAMES},
                      'Khía cạnh cảnh báo', 'tang1_khia_canh')

    if month == 'chuoi':
        filters = {**filters, 'months': chain_months}
        st.caption('Các tháng trong chuỗi · ' + ', '.join(map(thang_nam, chain_months)))
    elif month:
        filters = {**filters, 'months': (month,)}

    alerts = content_scope(ctx, filters)
    if alerts.empty:
        st.info('Không có bình luận cảnh báo trong phạm vi đang xem.')
        return

    incidents = alerts.loc[alerts.record_type.eq('a8_incident')].set_index('review_id')
    reflections = alerts.loc[alerts.record_type.eq('aspect_reflection')]
    filtered_reflections = reflections.loc[reflections.aspect.eq(aspect)] if aspect else reflections
    review_ids = incidents.index.union(filtered_reflections.review_id)
    matches = reviews.loc[reviews.review_id.isin(review_ids)].assign(
        has_a8=lambda rows: rows.review_id.isin(incidents.index))
    matches = matches.sort_values(['has_a8', 'review_date_iso', 'review_id'],
                                  ascending=[False, False, True])
    if matches.empty:
        st.info('Không có bình luận cảnh báo phù hợp với lựa chọn.')
        return

    counts = reflections.groupby('review_id').size()
    labels = {}
    for row in matches.itertuples():
        tags = [':red[**Sự cố bất thường · A8**]'] if row.has_a8 else []
        count = counts.get(row.review_id, 0)
        if count:
            tags.append(f'{count} khía cạnh')
        labels[row.review_id] = (
            f'**{nhan_van_ban(row.review_id)}** · {thang_nam(row.review_month)}  \n'
            f'{nhan_van_ban(van_ban_ngan(row.review_text))}  \n' + ' · '.join(tags)
        )

    left, right = st.columns([1, 1.8])
    review_key = f'tang1_review_{restaurant_id}_{month or "all"}_{aspect or "all"}'
    # Nút Tổng quan ưu tiên mở đúng mục mà người dùng vừa chọn.
    focus = st.session_state.pop('tang1_uu_tien', None)
    if focus:
        focus_ids = incidents.index if focus == 'a8' else reflections.review_id
        matches_focus = matches.loc[matches.review_id.isin(focus_ids), 'review_id']
        if not matches_focus.empty:
            preferred = matches_focus.iloc[0]
            st.session_state[review_key] = preferred
            st.session_state[review_key + '_trang'] = list(labels).index(preferred) // 4 + 1

    with left.container(border=True, key='danh_sach_canh_bao'):
        st.markdown('#### Bình luận có cảnh báo')
        st.caption(f'{len(matches):,} bình luận · A8 được ưu tiên lên đầu.')
        if aspect:
            st.caption('Sự cố A8 vẫn được hiển thị khi lọc khía cạnh.')
        review_id = chon_binh_luan(labels, review_key)

    with right.container(border=True, key='chi_tiet_canh_bao'):
        review = matches.loc[matches.review_id.eq(review_id)].iloc[0]
        levels = reflections.loc[reflections.review_id.eq(review_id)].set_index('aspect').severity
        aspects = ctx['aspects']
        aspects = aspects.loc[aspects.review_id.eq(review_id) & aspects.aspect.isin(levels.index)]
        incident = incidents.loc[review_id] if review_id in incidents.index else None
        problems = ctx['problems']
        levels_review = problems.loc[problems.review_id.eq(review_id), 'problem_level']
        muc_do = levels_review.iloc[0] if incident is None and not levels_review.empty else None
        hien_binh_luan(review, muc_do, aspects, levels, incident)
