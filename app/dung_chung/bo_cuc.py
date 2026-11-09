"""Bố cục và các thành phần dùng chung của giao diện."""
from html import escape

import streamlit as st
def khoi_tao(app):
    st.set_page_config(page_title='Cảnh báo nhà hàng',layout='wide')
    css=(app/'tai_nguyen/giao_dien.css').read_text(encoding='utf-8')
    st.markdown('<style>' + css + '</style>', unsafe_allow_html=True)
    st.title('Cảnh báo nhà hàng')
    st.caption('Xem phản ánh cần chú ý và xu hướng theo tháng của nhà hàng đang chọn.')
def tieu_de(ten,mo_ta):
    st.subheader(ten);st.caption(mo_ta)
def chi_so(values):
    for col,(name,value) in zip(st.columns(len(values)),values.items()):
        col.metric(name,'—' if value is None else f'{value:,}' if isinstance(value,int) else str(value))
def bang(rows,key,limit=30):
    if rows.empty:
        st.info('Không có dữ liệu phù hợp với lựa chọn.')
        return
    count=max(1,(len(rows)+limit-1)//limit)
    name=key+'_trang'
    if st.session_state.get(name,1)>count:st.session_state[name]=1
    page=int(st.number_input('Trang',min_value=1,max_value=count,value=1,key=name)) if count>1 else 1
    st.caption(f'{len(rows):,} dòng · Trang {page}/{count}')
    st.dataframe(rows.iloc[(page-1)*limit:page*limit],hide_index=True,use_container_width=True)
def chon(options,labels,label,key):
    options=list(options)
    if not options:return None
    if st.session_state.get(key) not in options:st.session_state[key]=options[0]
    return st.selectbox(label,options,format_func=lambda x:labels.get(x,str(x)),key=key)


def thang_nam(month):
    return f'{month[5:]}/{month[:4]}'


def huy_hieu(text, tone):
    return f'<strong class="huy-hieu huy-hieu-{escape(tone)}">{escape(text)}</strong>'
