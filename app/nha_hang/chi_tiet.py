"""Thông tin và nhận xét của một nhà hàng."""
import pandas as pd
import streamlit as st

from dung_chung.bo_cuc import bang, chi_so, chon
from ket_noi.trang_thai import ASPECT_NAMES, NHAN
from nhan_xet.chi_tiet import hien_chi_tiet as hien_nhan_xet


def hien_chi_tiet(ctx, reviews):
    reviews = reviews.sort_values('review_date_iso', ascending=False)
    latest = reviews.iloc[0]
    st.header(latest['restaurant_name'])
    st.caption(latest['restaurant_id'])

    info = []
    for label, options in [('Thành phố', ('city_std', 'city')), ('Tỉnh / thành', ('province',)),
                           ('Nguồn dữ liệu', ('source',)), ('Mã nhà hàng tại nguồn', ('source_restaurant_id',))]:
        for column in options:
            if column in reviews:
                values = reviews[column].dropna().astype(str).str.strip()
                if values.ne('').any():
                    info.append(f"**{label}** · {', '.join(values[values.ne('')].unique())}")
                    break
    for line in info:
        st.write(line)

    metrics = {'Nhận xét của nhà hàng': len(reviews),
               'Tháng có nhận xét': int(reviews.review_month.nunique())}
    if 'rating_numeric' in reviews:
        ratings = pd.to_numeric(reviews.rating_numeric, errors='coerce').dropna()
        if not ratings.empty:
            metrics['Điểm đánh giá trung bình'] = f'{ratings.mean():.1f}'
    chi_so(metrics)
    first, last = reviews.review_date_iso.iloc[-1][:10], latest['review_date_iso'][:10]
    st.caption(f'Nhận xét được lưu từ {first} đến {last}. Trang này xem toàn bộ thời gian của nhà hàng.')

    with st.expander('Phân bố phản hồi theo khía cạnh'):
        aspects = ctx['aspects']
        related = aspects.loc[aspects.review_id.isin(reviews.review_id)]
        counts = (pd.crosstab(related.aspect, related.label)
                  .reindex(index=ASPECT_NAMES, columns=NHAN, fill_value=0)
                  .rename(index=ASPECT_NAMES, columns=NHAN))
        st.caption('Số nhận xét theo từng nhãn cảm xúc của mỗi khía cạnh.')
        st.dataframe(counts.rename_axis('Khía cạnh').reset_index(), hide_index=True)

    st.subheader('Nhận xét của nhà hàng')
    text = st.text_input('Tìm mã hoặc nội dung nhận xét', key='nha_hang_tim').strip()
    if text:
        reviews = reviews.loc[reviews.review_id.str.contains(text, case=False, regex=False)
                              | reviews.review_text.str.contains(text, case=False, regex=False)]
    columns = {'review_id': 'Mã nhận xét', 'review_date_iso': 'Ngày đăng',
               'rating': 'Điểm đánh giá', 'review_text': 'Nội dung'}
    table = reviews[[key for key in columns if key in reviews]].rename(columns=columns)
    table['Ngày đăng'] = table['Ngày đăng'].str[:10]
    bang(table, 'nha_hang_nhan_xet')
    labels = {row.review_id: f'{row.review_date_iso[:10]} · {row.review_id} · {row.review_text[:80]}'
              for row in reviews.itertuples()}
    selected = chon(reviews.review_id, labels, 'Chọn nhận xét để đọc đầy đủ', 'nha_hang_chon_nhan_xet')
    hien_nhan_xet(ctx, selected, key='nha_hang')
