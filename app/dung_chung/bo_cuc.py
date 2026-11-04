"""Bố cục, điều hướng và thành phần trình bày dùng chung."""
import streamlit as st

def khoi_tao(app):
    st.set_page_config(page_title='Quản lý nhà hàng', layout='wide')
    css = (app / 'tai_nguyen/giao_dien.css').read_text(encoding='utf-8')
    st.markdown('<style>'+css+'</style>', unsafe_allow_html=True)
    st.title('Quản lý nhà hàng')
    st.caption('Phân tích phản hồi theo khía cạnh và theo dõi các vấn đề cần chú ý')
    st.info('Bản dựng giao diện · Chưa kết nối dữ liệu, mô hình hoặc hai tầng cảnh báo.')

def dieu_huong():
    with st.sidebar:
        st.markdown('### Restaurant Quality')
        st.caption('Không gian theo dõi phản hồi khách hàng')
        trang = st.radio('Điều hướng', ['Tổng quan', 'Nhà hàng', 'Nhận xét', 'Cảnh báo tầng 1', 'Cảnh báo tầng 2', 'Thông tin dữ liệu'], key='dieu_huong')
        st.divider()
    return trang

def tieu_de(ten, mo_ta):
    st.subheader(ten)
    st.caption(mo_ta)

def trong(ten, mo_ta):
    with st.container(border=True):
        st.markdown('#### '+ten)
        st.write(mo_ta)
        st.caption('Chưa kết nối dữ liệu')

def chi_so(cac_ten):
    for cot, ten in zip(st.columns(len(cac_ten)), cac_ten):
        cot.metric(ten, '—')

def bang(cot):
    # Chỉ dựng tiêu đề bảng, không tạo hàng dữ liệu minh họa.
    st.table({ten: [] for ten in cot})
    st.caption('Chưa có dữ liệu để hiển thị; dấu — không đại diện cho giá trị 0.')
