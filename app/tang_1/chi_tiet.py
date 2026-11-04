import streamlit as st
from dung_chung.bo_cuc import trong

def hien_chi_tiet():
    trong('Chi tiết phản ánh tầng 1', 'Vị trí hiển thị khía cạnh có vấn đề, mức độ và căn cứ từ nhận xét.')
    st.caption('Khi có A8, khu vực này hiển thị sự cố nghiêm trọng và ẩn mức độ chung Nhẹ/Vừa/Nặng. Chưa áp dụng phân loại ở bản giao diện này.')
    with st.expander('Nhận xét gốc và phân tích đầy đủ'):
        hien_chi_tiet_nhan_xet()

# Đặt tên riêng để tránh nhầm hàm chi tiết cảnh báo và chi tiết nhận xét.
from nhan_xet.chi_tiet import hien_chi_tiet as hien_chi_tiet_nhan_xet
