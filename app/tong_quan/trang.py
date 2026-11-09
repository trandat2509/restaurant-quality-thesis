"""Tóm tắt cảnh báo trong các chuỗi xu hướng của nhà hàng đang chọn."""
from html import escape

import streamlit as st

from dung_chung.bo_cuc import huy_hieu, thang_nam
from ket_noi.trang_thai import ASPECT_NAMES, content_scope, master_scope, monthly_scope, require_part
from src.alerts.tang2.view_trend_alerts import alert_description, comparison_months, empty_message
from src.alerts.view_content_alerts import ten_su_co
from tang_1.danh_sach import KHIA_CANH, SU_CO
from tang_2.chi_tiet import hien_bieu_do


def hien_trang(ctx, filters, chuyen_muc):
    restaurant = filters['restaurant']
    shop = master_scope(ctx, {'restaurant': restaurant}).iloc[0]
    st.header('Tổng quan')
    st.subheader(shop['restaurant_name'])
    month = filters['start']
    st.caption(f'Tháng phát cảnh báo · {thang_nam(month)}' if month
               else 'Tháng phát cảnh báo · Tất cả')

    trends = monthly_scope(ctx, filters)
    chains = [comparison_months(row) for row in trends.to_dict('records')]
    months = tuple(sorted({item['month'] for chain in chains for item in chain}))
    scope = (restaurant, months)
    with st.container(border=True, key='tong_quan_xu_huong'):
        st.markdown('### Diễn biến phản ánh tiêu cực')
        actions = st
        if require_part(ctx, 't2'):
            if not chains:
                coverage = monthly_scope(ctx, filters, coverage=True)
                st.info(empty_message(coverage.to_dict('records')))
            else:
                row = trends.iloc[0]
                summary, chart = st.columns([2, 3])
                actions = summary
                with summary:
                    st.markdown(f'<div class="dau-the"><strong>{escape(row["aspect_name"])}</strong>'
                                f'{huy_hieu(row["level_title"], row["level"])}</div>', unsafe_allow_html=True)
                    st.write(alert_description(row))
                    if len(chains) > 1:
                        st.caption(f'Đang hiển thị 1 trong {len(chains)} chuỗi cảnh báo.')
                with chart:
                    hien_bieu_do(chains[0], row['baseline_rate'])
        actions.button('Xem chi tiết', key='tong_quan_xem_xu_huong', on_click=chuyen_muc,
                       args=('Cảnh báo tầng 2',))

    st.caption('Hai khung dưới tổng hợp các tháng thuộc chuỗi xu hướng'
               + (' · ' + ', '.join(map(thang_nam, months)) if months else '.'))
    alerts = content_scope(ctx, {'restaurant': restaurant, 'months': months})
    with st.container(key='tong_quan_bo_sung'):
        left, right = st.columns(2)
        with left.container(border=True, key='tong_quan_khia_canh'):
            st.markdown('### Phản ánh theo khía cạnh')
            if require_part(ctx, 't1'):
                if not months:
                    st.info('Chưa có chuỗi xu hướng để tổng hợp.')
                else:
                    rows = alerts.loc[alerts.record_type.eq('aspect_reflection')]
                    if rows.empty:
                        st.info('Không có phản ánh theo khía cạnh trong các chuỗi đang xét.')
                    else:
                        st.metric('Bình luận có phản ánh', rows.review_id.nunique())
                        counts = rows.groupby('aspect').review_id.nunique().sort_values(ascending=False)
                        for aspect, count in counts.head(3).items():
                            st.write(f'{ASPECT_NAMES[aspect]} · {count} bình luận')
                        if len(counts) > 3:
                            st.caption(f'Còn {len(counts) - 3} khía cạnh trong trang chi tiết.')
            st.button('Xem chi tiết', key='tong_quan_xem_khia_canh', on_click=chuyen_muc,
                      args=('Cảnh báo tầng 1', KHIA_CANH, scope))

        with right.container(border=True, key='tong_quan_a8'):
            st.markdown('### Sự cố bất thường')
            if require_part(ctx, 't1'):
                if not months:
                    st.info('Chưa có chuỗi xu hướng để tổng hợp.')
                else:
                    incidents = alerts.loc[alerts.record_type.eq('a8_incident')]
                    if incidents.empty:
                        st.info('Không có sự cố bất thường xảy ra.')
                    else:
                        confirmed = int(incidents.review_status.eq('DA_DUYET').sum())
                        st.metric('Bình luận ghi nhận sự cố', incidents.review_id.nunique())
                        st.write(f'{confirmed} đã xác nhận · {len(incidents) - confirmed} cần xem xét')
                        names = sorted({name for codes in incidents.incident_types for name in ten_su_co(codes)})
                        st.caption(' · '.join(names))
            st.button('Xem chi tiết', key='tong_quan_xem_a8', on_click=chuyen_muc,
                      args=('Cảnh báo tầng 1', SU_CO, scope))
