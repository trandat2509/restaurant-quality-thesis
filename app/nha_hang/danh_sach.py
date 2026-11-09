"""Trang nhà hàng theo phạm vi đang chọn."""
from ket_noi.trang_thai import master_scope
from nha_hang.chi_tiet import hien_chi_tiet


def hien_trang(ctx, filters):
    reviews = master_scope(ctx, {'restaurant': filters['restaurant']})
    hien_chi_tiet(ctx, reviews)
