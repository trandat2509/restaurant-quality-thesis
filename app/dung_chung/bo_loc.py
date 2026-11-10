"""Chọn nhà hàng và tháng phát cảnh báo."""
import streamlit as st

from dung_chung.bo_cuc import chon, thang_nam


def chon_nha_hang(ctx):
    problems, trends = ctx['problems'], ctx['trend']
    tier2_ids = set(trends.restaurant_id) if not trends.empty else set()
    tier1_ids = set(problems.restaurant_id) if not problems.empty else set()
    shops = (ctx['master'][['restaurant_id', 'restaurant_name']]
             .drop_duplicates('restaurant_id').sort_values('restaurant_name'))
    labels = {row.restaurant_id: f'{row.restaurant_name} · {row.restaurant_id}'
              for row in shops.itertuples()}
    options = sorted(labels, key=lambda rid: (rid not in tier2_ids, rid not in tier1_ids))
    return chon(options, labels, 'Chọn nhà hàng', 'chon_nha_hang')


def hien_bo_loc(ctx, restaurant_id):
    master = ctx['master']
    months = master.loc[master.restaurant_id.eq(restaurant_id), 'review_month'].unique()
    labels = {'': 'Tất cả các tháng', **{month: thang_nam(month) for month in months}}
    month = chon(['', *sorted(months, reverse=True)], labels,
                 'Tháng phát cảnh báo', 'loc_thang_canh_bao')
    return {'restaurant': restaurant_id, 'start': month, 'end': month}
