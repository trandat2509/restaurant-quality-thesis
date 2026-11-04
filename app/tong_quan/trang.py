import streamlit as st
from dung_chung.bo_cuc import tieu_de, chi_so, trong, bang

def hien_trang():
    tieu_de('Tổng quan', 'Nhìn nhanh quy mô dữ liệu và các nhóm thông tin cần theo dõi.')
    chi_so(['Nhà hàng', 'Nhận xét', 'Phản ánh tầng 1', 'Phản ánh A8', 'Nhà hàng có cảnh báo xu hướng'])
    a,b=st.columns([2,1])
    with a:
        trong('Nhận xét theo thời gian', 'Vị trí biểu đồ số nhận xét theo tháng.')
    with b:
        trong('Phản hồi theo khía cạnh', 'Vị trí phân bố cảm xúc theo A1–A7.')
    st.markdown('#### Nhà hàng cần chú ý')
    bang(['Nhà hàng','Nhận xét','Phản ánh tầng 1','Phản ánh A8','Cảnh báo xu hướng'])
    with st.expander('Cách đi đến thông tin chi tiết'):
        st.write('Chọn Nhà hàng trên thanh điều hướng để mở bố cục chi tiết. Chọn từng tầng cảnh báo để xem bố cục cảnh báo và đường đi đến nhận xét.')
