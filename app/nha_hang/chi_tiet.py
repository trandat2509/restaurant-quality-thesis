"""Thông tin nhà hàng và nhận xét gốc."""
import pandas as pd
import streamlit as st

from dung_chung.bo_cuc import chon, chon_binh_luan, nhan_van_ban, thang_nam, the_so, van_ban_ngan
from nhan_xet.chi_tiet import hien_binh_luan_goc


def hien_chi_tiet(ctx, reviews):
    if reviews.empty:
        st.info('Chưa có nhận xét của nhà hàng này.')
        return

    reviews = reviews.sort_values('review_date_iso', ascending=False)
    latest = reviews.iloc[0]
    months = sorted(reviews.review_month.dropna().unique(), reverse=True)
    with st.container(border=True, key='ho_so_nha_hang'):
        st.subheader(latest['restaurant_name'])
        info = [latest['restaurant_id']]
        for label, options in [('Thành phố', ('city_std', 'city')),
                               ('Tỉnh / thành', ('province',)),
                               ('Nguồn nhận xét', ('source',)),
                               ('Mã nhà hàng tại nguồn', ('source_restaurant_id',))]:
            for column in options:
                if column in reviews:
                    values = reviews[column].dropna().astype(str).str.strip()
                    if values.ne('').any():
                        info.append(f"{label} {', '.join(values[values.ne('')].unique())}")
                        break
        st.text(' · '.join(info))

        metrics = [('Tổng số nhận xét', f'{len(reviews):,}'), ('Số tháng có nhận xét', len(months))]
        if 'rating_numeric' in reviews:
            ratings = pd.to_numeric(reviews.rating_numeric, errors='coerce').dropna()
            if not ratings.empty:
                metrics.append(('Điểm đánh giá trung bình', f'{ratings.mean():.1f}'))
        for col, (name, value) in zip(st.columns(len(metrics)), metrics):
            with col:
                the_so(name, value, mau='teal')
        if months:
            st.caption(f'Dữ liệu nhận xét từ {thang_nam(months[-1])} đến {thang_nam(months[0])}.')

    st.subheader('Nhận xét của nhà hàng')
    st.caption('Tìm theo nội dung hoặc thời gian rồi chọn nhận xét để đọc đầy đủ.')
    left, right = st.columns([1, 1.7])
    key = f'nha_hang_{latest["restaurant_id"]}'
    with left.container(border=True, key='danh_sach_nha_hang'):
        st.markdown('#### Danh sách nhận xét')
        query = st.text_input('Tìm mã hoặc nội dung nhận xét', key=key + '_tim').strip()
        month = chon(['', *months], {'': 'Tất cả các tháng', **{m: thang_nam(m) for m in months}},
                     'Thời gian', key + '_thang')
        selected_rows = reviews
        if month:
            selected_rows = selected_rows.loc[selected_rows.review_month.eq(month)]
        if query:
            selected_rows = selected_rows.loc[
                selected_rows.review_id.str.contains(query, case=False, regex=False, na=False)
                | selected_rows.review_text.str.contains(query, case=False, regex=False, na=False)]
        labels = {
            row.review_id: (f'**{nhan_van_ban(row.review_id)}** · {thang_nam(row.review_month)}  \n'
                            f'{nhan_van_ban(van_ban_ngan(row.review_text))}')
            for row in selected_rows.itertuples()
        }
        selected = chon_binh_luan(labels, key + '_review')
        if not selected:
            st.info('Không có nhận xét phù hợp.')

    if selected:
        with right.container(border=True, key='chi_tiet_nha_hang'):
            st.markdown('#### Chi tiết nhận xét')
            st.text(selected)
            hien_binh_luan_goc(selected_rows.loc[selected_rows.review_id.eq(selected)].iloc[0])
