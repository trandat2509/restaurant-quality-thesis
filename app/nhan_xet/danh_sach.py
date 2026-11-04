import streamlit as st
from dung_chung.bo_cuc import tieu_de, bang
from nhan_xet.chi_tiet import hien_chi_tiet

def hien_trang():
    tieu_de('Tra cứu nhận xét', 'Tra cứu toàn bộ nhận xét, kể cả những nhận xét không phát sinh cảnh báo.')
    st.text_input('Tìm trong nội dung nhận xét', placeholder='Chưa kết nối dữ liệu', disabled=True)
    bang(['Ngày đăng','Nhà hàng','Nội dung','Khía cạnh','Thao tác'])
    with st.expander('Xem bố cục chi tiết nhận xét'):
        hien_chi_tiet()
