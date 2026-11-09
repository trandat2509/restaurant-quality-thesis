import streamlit as st
from dung_chung.bo_cuc import tieu_de,bang,chon
from ket_noi.trang_thai import require_part,master_scope
from nhan_xet.chi_tiet import hien_chi_tiet
def hien_trang(ctx,filters):
    tieu_de('Tra cứu nhận xét','Tra cứu nhận xét gốc, nhãn ưu tiên và bằng chứng của từng nguồn.')
    if not require_part(ctx,'system'):return
    rows=master_scope(ctx,filters)
    q=st.text_input('Tìm mã hoặc nội dung nhận xét',key='tim_nhan_xet').strip()
    if q:rows=rows[rows.review_text.str.contains(q,case=False,regex=False)|rows.review_id.str.contains(q,case=False,regex=False)]
    rows=rows.sort_values(['review_date_iso','review_id'],ascending=False)
    bang(rows[['review_id','review_date_iso','restaurant_name','review_text']].rename(columns={
        'review_id':'Mã nhận xét','review_date_iso':'Ngày đăng','restaurant_name':'Nhà hàng','review_text':'Nội dung'}),'ds_nhan_xet')
    ids=rows.review_id.tolist()
    labels={r.review_id:r.review_id+' · '+r.review_text[:65] for r in rows.itertuples()}
    selected=chon(ids,labels,'Chọn nhận xét để xem đầy đủ','chon_nhan_xet')
    hien_chi_tiet(ctx,selected)
