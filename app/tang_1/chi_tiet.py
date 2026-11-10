"""Chi tiết nhận xét và tất cả cảnh báo thuộc nhận xét đó."""
import streamlit as st

from dung_chung.bo_cuc import huy_hieu, thang_nam
from nhan_xet.chi_tiet import hien_binh_luan_goc, hien_khia_canh, hien_su_co


def hien_binh_luan(review, muc_do, aspects, levels, incident=None):
    st.markdown('#### Chi tiết bình luận')
    st.text(f"{review['review_id']} · {thang_nam(review['review_month'])}")
    if incident is not None:
        st.markdown(huy_hieu('Sự cố nghiêm trọng', 'red'), unsafe_allow_html=True)
        hien_su_co(review, incident['incident_types'], incident['review_status'],
                   incident['evidence_text'], incident['evidence_needs_review'], show_review=False)
    hien_binh_luan_goc(review, muc_do)
    if not aspects.empty:
        with st.container(border=True, key='tang1_khia_canh_noi_dung'):
            st.markdown('#### Cảnh báo theo khía cạnh')
            hien_khia_canh(aspects, levels)
