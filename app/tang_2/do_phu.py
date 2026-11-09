import streamlit as st
from dung_chung.bo_cuc import bang
from ket_noi.trang_thai import monthly_scope
def hien_do_phu(ctx,filters,key='do_phu'):
    rows=monthly_scope(ctx,filters,coverage=True)
    st.caption('Độ phủ phản ánh khả năng đánh giá, không đồng nghĩa có cảnh báo.')
    cols=['restaurant_name','month','baseline_months','review_count','text_group_count','status_title','reason']
    bang(rows.reindex(columns=cols).rename(columns={'restaurant_name':'Nhà hàng','month':'Tháng',
        'baseline_months':'Cơ sở (tháng)','review_count':'Nhận xét','text_group_count':'Nhóm văn bản',
        'status_title':'Trạng thái','reason':'Lý do'}),key)
