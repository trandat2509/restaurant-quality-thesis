import streamlit as st
from dung_chung.bo_cuc import trong, bang

def hien_chi_tiet():
    st.markdown('#### Chi tiết nhận xét')
    st.caption('Mã nhận xét: — · Nhà hàng: — · Ngày đăng: —')
    trong('Nhận xét nguyên văn', 'Nội dung gốc sẽ hiển thị tại đây sau khi kết nối.')
    st.markdown('##### Phân tích theo khía cạnh')
    bang(['Khía cạnh','Nhãn cảm xúc','Bằng chứng','Nguồn nhãn'])
    with st.expander('Thông tin sự cố A8'):
        st.write('Nơi hiển thị loại sự cố, trạng thái duyệt và bằng chứng nếu có. Hiện chưa có dữ liệu để xác định A8.')
