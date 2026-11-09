"""Chạy từ thư mục gốc dự án: python -m streamlit run app/main.py."""
from pathlib import Path
import sys

APP = Path(__file__).resolve().parent
ROOT = APP.parent
for path in (ROOT, APP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st

from dung_chung.bo_cuc import khoi_tao
from dung_chung.bo_loc import chon_nha_hang, hien_bo_loc
from ket_noi.trang_thai import _nap, nap_du_lieu, require_part
from nha_hang.danh_sach import hien_trang as hien_nha_hang
from tang_1.danh_sach import hien_trang as hien_tang_1
from tang_2.danh_sach import hien_trang as hien_tang_2
from tong_quan.trang import hien_trang as hien_tong_quan

TRANG = {
    'Tổng quan': hien_tong_quan,
    'Nhà hàng': hien_nha_hang,
    'Cảnh báo tầng 1': hien_tang_1,
    'Cảnh báo tầng 2': hien_tang_2,
}


def dat_lai_chuoi():
    st.session_state.pop('tang1_chuoi', None)
    st.session_state.pop('tang1_thang', None)


def chuyen_muc(page, section=None, scope=None):
    st.session_state['dieu_huong'] = page
    if section is not None:
        st.session_state['tang1_muc'] = section
        st.session_state['tang1_chuoi'] = scope
        st.session_state['tang1_thang'] = 'chuoi'


khoi_tao(APP)
page = st.sidebar.radio('Điều hướng', list(TRANG), key='dieu_huong', on_change=dat_lai_chuoi)
if st.sidebar.button('Làm mới dữ liệu'):
    _nap.clear()
    st.rerun()

ctx = nap_du_lieu(ROOT)
if not require_part(ctx, 'system'):
    st.stop()

with st.sidebar:
    st.markdown('#### Phạm vi xem')
    restaurant_id = chon_nha_hang(ctx)
    if restaurant_id is None:
        st.info('Chưa có nhà hàng trong dữ liệu System.')
        st.stop()
    filters = (hien_bo_loc(ctx, restaurant_id) if page in ('Tổng quan', 'Cảnh báo tầng 2')
               else {'restaurant': restaurant_id})

if page == 'Tổng quan':
    hien_tong_quan(ctx, filters, chuyen_muc)
else:
    TRANG[page](ctx, filters)
