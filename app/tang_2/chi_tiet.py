import streamlit as st
from dung_chung.bo_cuc import trong, bang
from nhan_xet.chi_tiet import hien_chi_tiet as hien_nhan_xet

def hien_chi_tiet():
    trong('Chi tiết cảnh báo xu hướng', 'Nhà hàng: — · Tháng: — · Khía cạnh: — · Mức cảnh báo: —')
    bang(['Kỳ đối chiếu','Số nhận xét','Số tiêu cực','Tỷ lệ'])
    st.caption('Số kỳ đối chiếu và ngưỡng sẽ lấy từ kết quả hệ thống khi kết nối, không cố định trong giao diện.')
    with st.expander('Nhận xét liên quan → Phân tích đầy đủ'):
        hien_nhan_xet()
