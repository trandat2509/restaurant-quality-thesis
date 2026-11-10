"""Bố cục và các thành phần dùng chung của giao diện."""
from html import escape
import re

import streamlit as st
def khoi_tao(app):
    st.set_page_config(page_title='Cảnh báo nhà hàng',layout='wide')
    css=(app/'tai_nguyen/giao_dien.css').read_text(encoding='utf-8')
    st.markdown('<style>' + css + '</style>', unsafe_allow_html=True)



def tieu_de(ten, mo_ta):
    st.markdown('<div class="nhan-trang">Giám sát chất lượng dịch vụ</div>',
                unsafe_allow_html=True)
    st.title(ten)
    st.caption(mo_ta)
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



def the_so(ten, gia_tri, ghi_chu='', mau=''):
    tone = escape(mau, quote=True)
    note = f'<small>{escape(str(ghi_chu))}</small>' if ghi_chu else ''
    value = '—' if gia_tri is None else str(gia_tri)
    st.markdown(f'<div class="the-so the-so-{tone}"><span>{escape(ten)}</span>'
                f'<div><strong>{escape(value)}</strong>{note}</div></div>',
                unsafe_allow_html=True)


def van_ban_ngan(value, limit=100):
    text = ' '.join(str(value).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def nhan_van_ban(value):
    return re.sub(r'([\\`*_{}\[\]()<>!|$:#~^])', r'\\\1', str(value))


def chon_binh_luan(labels, key, limit=4):
    if not labels:
        return None

    options = list(labels)
    pages = (len(options) + limit - 1) // limit
    location = st.empty()
    page_key = f'{key}_trang'
    if st.session_state.get(page_key, 1) > pages:
        st.session_state[page_key] = 1
    if pages > 1:
        page = int(st.number_input('Trang bình luận', 1, pages, key=page_key))
    else:
        page = 1

    offset = (page - 1) * limit
    items = options[offset:offset + limit]
    with location.container():
        if st.session_state.get(key) not in items:
            st.session_state[key] = items[0]
        selected = st.radio('Chọn bình luận', items, format_func=labels.__getitem__,
                            key=key, label_visibility='collapsed')
    st.caption(f'{offset + 1}–{offset + len(items)} / {len(options):,} bình luận')
    return selected
