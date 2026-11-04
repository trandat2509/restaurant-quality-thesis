import streamlit as st
from dung_chung.bo_cuc import tieu_de, chi_so, bang
from tang_1.chi_tiet import hien_chi_tiet

def hien_noi_dung():
    st.selectbox('Loại phản ánh', ['Tất cả phản ánh'], disabled=True)
    bang(['Nhà hàng','Ngày đăng','Khía cạnh có vấn đề','Mức độ / A8','Thao tác'])
    with st.expander('Xem bố cục chi tiết phản ánh'):
        hien_chi_tiet()

def hien_trang():
    tieu_de('Cảnh báo tầng 1', 'Phản ánh theo nội dung của từng nhận xét.')
    chi_so(['Nhận xét có vấn đề','Phản ánh A8','Cần xem xét'])
    hien_noi_dung()
