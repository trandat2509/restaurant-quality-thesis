import streamlit as st
import pandas as pd
from ket_noi.trang_thai import TRANG_THAI
from src.modeling.phobert_system_common import load_settings
def hien_trang(ctx,filters):
    st.subheader('Thông tin dữ liệu')
    st.caption(TRANG_THAI)
    rows=[]
    for part,name in [('system','System'),('t1','Cảnh báo tầng 1'),('t2','Cảnh báo tầng 2')]:
        obj=ctx.get(part)
        manifest=(obj or {}) if part=='system' else obj.manifest if obj else {}
        rows.append({'Thành phần':name,'Trạng thái':'Sẵn sàng' if obj is not None else 'Chưa sẵn sàng',
                     'Phiên':manifest.get('current_model_version',manifest.get('version','')),
                     'Thời điểm':manifest.get('created_at_utc',''),'Thông báo':ctx['errors'].get(part,'')})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    with st.expander('Đường dẫn đang sử dụng',expanded=ctx['system'] is None):
        for part in ['system','t1','t2']:
            if part in ctx['errors']:st.error(part+': '+ctx['errors'][part])
        if ctx['system'] is not None:st.write('System:',str(ctx['paths']['directory']))
        if ctx['t1']:st.write('Tầng 1:',str(ctx['t1'].paths['data']))
        if ctx['t2']:st.write('Tầng 2:',str(ctx['t2'].directory))
        root=ctx['root']
        for name in ['aspect_predictions_with_evidence_add.csv','prediction_complete_add.json']:
            path=root/'data/predictions/foody_add'/name
            st.write(('Có tệp: ' if path.is_file() else 'Chưa có tệp: ')+str(path))
        st.caption('Có tệp dự đoán chưa đồng nghĩa System và hai tầng đã được xây dựng thành công.')
    with st.expander('Điều kiện và cấu hình đang áp dụng'):
        if ctx['t2']:
            st.json(ctx['t2'].policy)
        try:
            cfg=load_settings(project_root=ctx['root'])
            st.json(cfg['alert_policy']['content'])
        except (OSError,KeyError,ValueError) as exc:st.info(str(exc))
    st.caption('Sau khi cập nhật bằng notebook, bấm Làm mới dữ liệu. Giao diện chỉ đọc kết quả đã kiểm tra.')
