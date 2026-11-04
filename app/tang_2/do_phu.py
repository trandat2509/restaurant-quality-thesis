import streamlit as st
from dung_chung.bo_cuc import bang

def hien_do_phu():
    st.markdown('##### Độ phủ thời gian')
    bang(['Tháng','Số nhận xét','Số nhóm văn bản','Điều kiện cần','Trạng thái'])
    st.caption('Sau khi kết nối sẽ phân biệt: chưa có kết quả, chưa đủ dữ liệu, đủ dữ liệu nhưng chưa đạt ngưỡng và có cảnh báo. Hiện chưa xác định trạng thái nào.')
