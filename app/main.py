"""Khởi động giao diện: python -m streamlit run app/main.py."""
from pathlib import Path
import sys
APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
import streamlit as st
from dung_chung.bo_cuc import khoi_tao, dieu_huong
from dung_chung.bo_loc import hien_bo_loc
from tong_quan.trang import hien_trang as tong_quan
from nha_hang.danh_sach import hien_trang as nha_hang
from nhan_xet.danh_sach import hien_trang as nhan_xet
from tang_1.danh_sach import hien_trang as tang_1
from tang_2.danh_sach import hien_trang as tang_2
from thong_tin_du_lieu.trang import hien_trang as thong_tin

khoi_tao(APP)
trang = dieu_huong()
hien_bo_loc()
{
    'Tổng quan': tong_quan,
    'Nhà hàng': nha_hang,
    'Nhận xét': nhan_xet,
    'Cảnh báo tầng 1': tang_1,
    'Cảnh báo tầng 2': tang_2,
    'Thông tin dữ liệu': thong_tin,
}[trang]()
