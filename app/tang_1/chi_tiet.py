"""Chi tiết một cảnh báo tầng 1."""
from nhan_xet.chi_tiet import hien_binh_luan_goc, hien_khia_canh, hien_su_co as hien_the_su_co


def hien_binh_luan(review, problem, aspects, levels):
    muc_do = None if problem['kind'] == 'a8' else problem['problem_level']
    hien_binh_luan_goc(review, muc_do)
    hien_khia_canh(aspects, levels)


def hien_su_co(review, incident):
    hien_the_su_co(review, incident['incident_types'], incident['review_status'],
                   incident['evidence_text'], incident['evidence_needs_review'])
