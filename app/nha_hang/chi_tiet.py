import streamlit as st
from dung_chung.bo_cuc import chi_so, trong
from tang_1.danh_sach import hien_noi_dung as tang_1
from tang_2.danh_sach import hien_noi_dung as tang_2
from nhan_xet.chi_tiet import hien_chi_tiet as nhan_xet

def hien_chi_tiet():
    st.markdown('### Chi tiết nhà hàng')
    st.caption('Tên nhà hàng: — · Mã nhà hàng: — · Thời gian: —')
    tabs=st.tabs(['Tổng quan nhà hàng','Phản ánh từng nhận xét','Xu hướng theo thời gian','Toàn bộ nhận xét'])
    with tabs[0]:
        chi_so(['Nhận xét','Phản ánh tầng 1','Phản ánh A8'])
        trong('Cảm xúc theo khía cạnh', 'Vị trí biểu đồ A1–A7 của nhà hàng đang chọn.')
    with tabs[1]: tang_1()
    with tabs[2]: tang_2()
    with tabs[3]: nhan_xet()
