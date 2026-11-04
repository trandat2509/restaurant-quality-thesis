import streamlit as st
from dung_chung.bo_cuc import tieu_de, bang
from nha_hang.chi_tiet import hien_chi_tiet

def hien_trang():
    tieu_de('Nhà hàng', 'Tìm nhà hàng và đi sâu vào phản ánh, xu hướng, nhận xét gốc.')
    st.text_input('Tìm tên hoặc mã nhà hàng', placeholder='Chưa kết nối dữ liệu', disabled=True)
    bang(['Mã nhà hàng','Tên nhà hàng','Số nhận xét','Tầng 1','Tầng 2','Thao tác'])
    st.toggle('Mở bố cục chi tiết nhà hàng', key='mo_nha_hang')
    if st.session_state.get('mo_nha_hang'):
        hien_chi_tiet()
