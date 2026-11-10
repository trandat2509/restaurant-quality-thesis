"""Tổng quan một nhà hàng và các chuỗi cảnh báo đang xem."""
from html import escape

import streamlit as st

from dung_chung.bo_cuc import huy_hieu, thang_nam, the_so
from ket_noi.trang_thai import ASPECT_NAMES, content_scope, monthly_scope, require_part
from src.alerts.tang2.view_trend_alerts import comparison_months, empty_message, pct
from src.alerts.view_content_alerts import ten_su_co
from tang_2.chi_tiet import hien_bieu_do


def hien_trang(ctx, filters, chuyen_muc):
    restaurant = filters['restaurant']
    trends = monthly_scope(ctx, filters)
    chains = [comparison_months(row) for row in trends.to_dict('records')]
    months = tuple(sorted({item['month'] for chain in chains for item in chain}))
    scope = (restaurant, months)
    alerts = content_scope(ctx, {'restaurant': restaurant, 'months': months})
    reflections = incidents = alerts
    confirmed = pending = None
    if ctx.get('t1') is not None:
        reflections = alerts.loc[alerts.record_type.eq('aspect_reflection')]
        incidents = alerts.loc[alerts.record_type.eq('a8_incident')]
        confirmed = incidents.loc[incidents.review_status.eq('DA_DUYET'), 'review_id'].nunique()
        pending = incidents.loc[incidents.review_status.ne('DA_DUYET'), 'review_id'].nunique()

    left, middle, right = st.columns(3)
    with left:
        count = trends.aspect.nunique() if not trends.empty else 0
        the_so('Khía cạnh có cảnh báo xu hướng', count if ctx.get('t2') is not None else None,
               'Cảnh báo tầng 2', 'yellow')
    with middle:
        the_so('Sự cố đã xác nhận', confirmed, 'Cảnh báo tầng 1 · A8', 'red')
    with right:
        the_so('Sự cố cần xem xét', pending, 'Cảnh báo tầng 1 · A8')

    with st.container(border=True, key='tong_quan_xu_huong'):
        st.subheader('Diễn biến phản ánh tiêu cực')
        if require_part(ctx, 't2'):
            if not chains:
                coverage = monthly_scope(ctx, filters, coverage=True)
                st.info(empty_message(coverage.to_dict('records')))
            else:
                row = trends.iloc[0]
                st.markdown(f'<div class="dau-the"><strong>{escape(row["aspect_name"])}</strong>'
                            f'{huy_hieu(row["level_title"], row["level"])}</div>', unsafe_allow_html=True)
                st.caption(f"Tháng {thang_nam(row['month'])} so với {row['baseline_months']} tháng liền trước.")
                left, middle, right = st.columns(3)
                with left:
                    the_so('Tỷ lệ tháng cảnh báo', pct(row['complaint_rate']),
                           f"{row['count_NEG']}/{row['review_count']} nhận xét", 'teal')
                with middle:
                    the_so('Mốc tham chiếu', pct(row['baseline_rate']), f"{row['baseline_months']} tháng trước")
                with right:
                    the_so('Mức tăng', f"{float(row['increase_pp']):+.1f}", 'điểm phần trăm', row['level'])
                hien_bieu_do(chains[0], row['baseline_rate'], row['level'])
                target = (restaurant, row['aspect'], row['month'], int(row['baseline_months']))
                st.button('Xem cảnh báo tầng 2 →', key='tong_quan_xem_xu_huong',
                          on_click=chuyen_muc, args=('Cảnh báo tầng 2',), kwargs={'trend': target})

    if months:
        st.caption('Hai khung dưới tổng hợp các tháng thuộc chuỗi xu hướng · '
                   + ', '.join(map(thang_nam, months)))
    with st.container(key='tong_quan_bo_sung'):
        left, right = st.columns([1.12, 1])
        with left.container(border=True, key='tong_quan_khia_canh'):
            st.markdown('### Phản ánh theo khía cạnh')
            if require_part(ctx, 't1'):
                if not months:
                    st.info('Chưa có chuỗi xu hướng để tổng hợp.')
                elif reflections.empty:
                    st.info('Không có phản ánh theo khía cạnh trong các chuỗi đang xét.')
                else:
                    counts = reflections.groupby('aspect').review_id.nunique().nlargest(3)
                    maximum = int(counts.max())
                    bars = []
                    for aspect, count in counts.items():
                        name = escape(ASPECT_NAMES.get(aspect, aspect))
                        bars.append(f'<div class="hang-phan-anh"><span>{name}</span>'
                                    f'<div class="nen-thanh"><i style="width:{100 * count / maximum:.1f}%"></i></div>'
                                    f'<strong>{int(count):,}</strong></div>')
                    st.markdown('<div class="phan-anh">' + ''.join(bars) + '</div>',
                                unsafe_allow_html=True)
            st.button('Xem chi tiết →', key='tong_quan_xem_khia_canh', on_click=chuyen_muc,
                      args=('Cảnh báo tầng 1',), kwargs={'scope': scope, 'focus': 'aspect'})

        with right.container(border=True, key='tong_quan_a8'):
            st.markdown('### Sự cố bất thường')
            if require_part(ctx, 't1'):
                if not months:
                    st.info('Chưa có chuỗi xu hướng để tổng hợp.')
                elif incidents.empty:
                    st.info('Không có sự cố bất thường xảy ra.')
                else:
                    names = sorted({name for codes in incidents.incident_types for name in ten_su_co(codes)})
                    st.markdown(f'<div class="dong-a8"><strong>{incidents.review_id.nunique():,} bình luận '
                                f'ghi nhận sự cố</strong>{huy_hieu(f"{confirmed} đã xác nhận", "red")}</div>',
                                unsafe_allow_html=True)
                    st.caption(' · '.join(names))
            st.button('Xem chi tiết →', key='tong_quan_xem_a8', on_click=chuyen_muc,
                      args=('Cảnh báo tầng 1',), kwargs={'scope': scope, 'focus': 'a8'})
