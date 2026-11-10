"""Chạy từ thư mục gốc dự án: python -m streamlit run app/main.py."""
from pathlib import Path
import sys

APP = Path(__file__).resolve().parent
ROOT = APP.parent
for path in (ROOT, APP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st

from dung_chung.bo_cuc import khoi_tao, tieu_de
from dung_chung.bo_loc import chon_nha_hang, hien_bo_loc
from ket_noi.trang_thai import _nap, nap_du_lieu, require_part
from nha_hang.danh_sach import hien_trang as hien_nha_hang
from tai_khoan import yeu_cau_dang_nhap
from tang_1.danh_sach import hien_trang as hien_tang_1
from tang_2.danh_sach import hien_trang as hien_tang_2
from tong_quan.trang import hien_trang as hien_tong_quan

TRANG = {
    'Tổng quan': (hien_tong_quan, 'Theo dõi phản ánh và sự cố trong phạm vi đã chọn.'),
    'Nhà hàng': (hien_nha_hang, 'Xem thông tin nhà hàng và đọc nhận xét của khách hàng.'),
    'Cảnh báo tầng 1': (hien_tang_1, 'Xem những bình luận cần chú ý của nhà hàng đang chọn.'),
    'Cảnh báo tầng 2': (hien_tang_2, 'Theo dõi diễn biến phản ánh tiêu cực theo từng tháng.'),
}
TRANG_XU_HUONG = ('Tổng quan', 'Cảnh báo tầng 2')


def dat_lai_chuoi():
    for key in ('tang1_chuoi', 'tang1_thang', 'tang1_uu_tien', 'tang2_muc_tieu'):
        st.session_state.pop(key, None)


def chuyen_muc(page, scope=None, focus=None, trend=None):
    dat_lai_chuoi()
    st.session_state['dieu_huong'] = page
    if scope and scope[1]:
        st.session_state['tang1_chuoi'] = scope
        st.session_state['tang1_thang'] = 'chuoi'
        st.session_state['tang1_khia_canh'] = ''
    if focus:
        st.session_state['tang1_uu_tien'] = focus
    if trend:
        st.session_state['tang2_muc_tieu'] = trend


khoi_tao(APP)
menu = st.sidebar.empty()
yeu_cau_dang_nhap()
with menu.container():
    st.markdown('### Cảnh báo nhà hàng')
    st.caption('Chất lượng dịch vụ')
    page = st.radio('Điều hướng', list(TRANG), key='dieu_huong', on_change=dat_lai_chuoi)
    if st.button('Làm mới dữ liệu'):
        _nap.clear()
        st.rerun()

view, description = TRANG[page]
tieu_de(page, description)
ctx = nap_du_lieu(ROOT)
if not require_part(ctx, 'system'):
    st.stop()

# Hiện bộ lọc ở đầu trang để cùng nằm trong vùng xem với số liệu.
if page in TRANG_XU_HUONG:
    restaurant_col, month_col = st.columns([1.15, 1])
    with restaurant_col:
        restaurant_id = chon_nha_hang(ctx)
    if restaurant_id is not None:
        with month_col:
            filters = hien_bo_loc(ctx, restaurant_id)
elif page == 'Cảnh báo tầng 1':
    restaurant_col, *controls = st.columns([1.17, 0.94, 1])
    with restaurant_col:
        restaurant_id = chon_nha_hang(ctx)
    filters = {'restaurant': restaurant_id}
else:
    restaurant_id = chon_nha_hang(ctx)
    filters = {'restaurant': restaurant_id}

if restaurant_id is None:
    st.info('Chưa có nhà hàng trong dữ liệu System.')
    st.stop()

if page == 'Tổng quan':
    view(ctx, filters, chuyen_muc)
elif page == 'Cảnh báo tầng 1':
    view(ctx, filters, controls)
else:
    view(ctx, filters)
