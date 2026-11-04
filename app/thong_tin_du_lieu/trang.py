import streamlit as st
from dung_chung.bo_cuc import tieu_de, bang
from ket_noi.trang_thai import TRANG_THAI

def hien_trang():
    tieu_de('Thông tin dữ liệu', 'Nơi kiểm tra phiên dữ liệu, mô hình và điều kiện đang áp dụng sau khi kết nối.')
    st.write('**Trạng thái kết nối:**', TRANG_THAI)
    bang(['Thành phần','Phiên bản','Thời điểm cập nhật','Trạng thái'])
    with st.expander('Điều kiện và cấu hình hai tầng'):
        st.write('Chưa đọc cấu hình cảnh báo. Các ngưỡng và số kỳ đối chiếu sẽ được hiển thị từ hệ thống khi triển khai phần kết nối.')
