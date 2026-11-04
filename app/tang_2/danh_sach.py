import streamlit as st
from dung_chung.bo_cuc import tieu_de, chi_so, bang, trong
from tang_2.chi_tiet import hien_chi_tiet
from tang_2.do_phu import hien_do_phu

def hien_noi_dung():
    st.selectbox('Mức cảnh báo xu hướng', ['Tất cả mức'], disabled=True)
    trong('Xu hướng theo thời gian', 'Vị trí biểu đồ tỷ lệ tiêu cực theo khía cạnh; tháng thiếu dữ liệu sẽ được thể hiện riêng khi kết nối.')
    bang(['Nhà hàng','Tháng','Khía cạnh','Tỷ lệ hiện tại','Tỷ lệ đối chiếu','Mức tăng','Kết quả'])
    with st.expander('Xem bố cục chi tiết cảnh báo'):
        hien_chi_tiet()
    with st.expander('Xem độ phủ và điều kiện đánh giá'):
        hien_do_phu()

def hien_trang():
    tieu_de('Cảnh báo tầng 2', 'Theo dõi biến động phản hồi theo nhà hàng, thời gian và khía cạnh.')
    chi_so(['Nhà hàng có cảnh báo','Cảnh báo vàng','Cảnh báo đỏ'])
    hien_noi_dung()
