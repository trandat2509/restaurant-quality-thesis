"""Bộ lọc giữ vị trí thống nhất; chưa lấy danh sách giá trị từ dữ liệu."""
import streamlit as st

def hien_bo_loc():
    with st.sidebar:
        st.markdown('#### Phạm vi xem')
        st.selectbox('Nhà hàng', ['Chưa kết nối dữ liệu'], disabled=True, key='loc_nha_hang')
        st.selectbox('Khoảng thời gian', ['Chưa kết nối dữ liệu'], disabled=True, key='loc_thoi_gian')
        st.selectbox('Khía cạnh', ['Tất cả khía cạnh'], disabled=True, key='loc_khia_canh')
        st.caption('Bộ lọc sẽ được kích hoạt khi kết nối dữ liệu.')
