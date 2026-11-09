"""Đọc và trình bày cảnh báo tầng 1 theo nhận xét; không ghi dữ liệu hoặc chạy mô hình."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts.build_content_alerts import (
    ASPECT_NAMES, output_paths, read_json, require, sha, signature, verify_outputs,
)
from src.modeling.phobert_system_common import load_settings
from src.pipeline.prepare_system_data import csv_rows, load_system_inputs
from src.alerts.content_evidence import EvidenceSelector, score_review

TEN_NHAN = {"POS": "Tích cực", "NEU": "Trung tính", "NEG": "Tiêu cực", "MIX": "Hỗn hợp"}
TEN_MUC_DO = {"light": "Nhẹ", "serious": "Nghiêm trọng", "unknown": "Chưa xác định"}
TEN_TRANG_THAI = {"DA_DUYET": "ĐÃ XÁC NHẬN", "CAN_XEM_XET": "CẦN XEM XÉT"}
TEN_SU_CO = {
    "PHAN_BIET_DOI_XU": "Phân biệt đối xử",
    "SUC_KHOE": "Sức khỏe",
    "XUC_PHAM_NGHIEM_TRONG": "Xúc phạm nghiêm trọng",
    "AN_TOAN_TAI_CO_SO": "An toàn tại cơ sở",
    "TRAI_NGHIEM_CUC_TIEU_CUC": "Trải nghiệm cực tiêu cực",
    "BAO_LUC_DE_DOA": "Bạo lực/đe dọa",
}
TEN_VAN_DE = {"nhe": "Nhẹ", "vua": "Vừa", "nang": "Nặng"}
THU_TU_NHOM = (
    ("a8_incident", "DA_DUYET"), ("a8_incident", "CAN_XEM_XET"),
    ("review_problem", "nang"), ("review_problem", "vua"), ("review_problem", "nhe"),
)

TEN_BANG_CHUNG = {'confirmed_evidence': 'Bằng chứng đã xác nhận',
                 'evidence': 'Bằng chứng mô hình hoặc nguồn chưa xác nhận', 'context': 'Ngữ cảnh xem nhanh'}


def giu_bang_chung(text, evidence_text, spans_json, confirmed=False):
    """Giữ nguyên nội dung đã nhập, kể cả nhiều đoạn và vị trí còn mơ hồ."""
    spans = json.loads(spans_json)
    for s in spans:
        require(type(s['start']) is int and type(s['end']) is int and 0 <= s['start'] < s['end'] <= len(text)
                and text[s['start']:s['end']] == s['text'], 'Vị trí bằng chứng nguồn không khớp nhận xét.')
    raw = [s.strip() for s in evidence_text.split('||') if s.strip()]
    require(all(s in text for s in raw), 'Bằng chứng nguồn không nằm trong nhận xét.')
    located = {s['text'].strip() for s in spans}
    return {'kind': 'confirmed_evidence' if confirmed else 'evidence', 'spans': spans,
            'unlocated_fragments': [s for s in raw if s not in located], 'complete': not bool(raw) or bool(spans)}


def dung(value):
    return str(value).lower() in {"1", "true"}


def ngay_hien_thi(value):
    """Giữ ngày tại múi giờ đã lưu trong system, không chuyển theo máy đang chạy."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d/%m/%Y")


def ten_su_co(value):
    # Chỉ đổi tên các mã đã lưu; tuyệt đối không suy loại sự cố từ lời nhận xét.
    codes = list(dict.fromkeys(x.strip() for x in value.split("|") if x.strip()))
    return [TEN_SU_CO.get(x, "Loại chưa có tên hiển thị: " + x) for x in codes] or ["Chưa xác định loại"]


class BoDocTang1:
    """Kiểm tra bản kết quả đã xuất và nối các nguồn bằng review_id."""

    def __init__(self, project_root=ROOT, config=None, kiem_tra_day_du=False, *, system_inputs=None):
        self.cfg = load_settings(config=config, project_root=project_root)
        # Giao diện có thể dùng chung đầu vào System đã kiểm tra trong lần đọc này.
        self.inputs = system_inputs if system_inputs is not None else load_system_inputs(
            project_root=self.cfg["_root"], view="operational", config=self.cfg["_config"])
        self.cfg["_system_snapshot"] = self.inputs["paths"]["directory"]
        self.paths = output_paths(self.cfg)
        require(self.paths["manifest"].is_file(),
                "Chưa có kết quả tầng 1. Chạy notebook từ ô đầu hoặc build_content_alerts --use-system.")
        self.manifest = read_json(self.paths["manifest"])
        require(self.manifest.get("status") == "complete", "Kết quả tầng 1 chưa hoàn tất.")
        require(self.manifest.get("signature") == signature(self.cfg, self.inputs["manifest"]),
                "Mã, cấu hình hoặc system đã đổi. Chạy lại notebook từ ô đầu hoặc build_content_alerts --use-system.")
        for key in ("data", "report"):
            require(self.paths[key].is_file() and
                    sha(self.paths[key]) == self.manifest["output_sha256"][key],
                    "Tệp kết quả thiếu hoặc đã bị sửa: " + str(self.paths[key]))
        if kiem_tra_day_du:
            verify_outputs(self.cfg, self.inputs, signature(self.cfg, self.inputs["manifest"]))
        self.master = self.inputs["master"]
        self.a8 = {x["review_id"]: x for x in self.inputs["a8"]}
        self.label_counts = defaultdict(Counter)
        for row in self.inputs["aspects"]:
            self.label_counts[row["review_id"]][row["label"]] += 1
        require(set(self.label_counts) == set(self.master) and
                all(sum(c.values()) == 7 for c in self.label_counts.values()),
                "System phải có đủ bảy nhãn cho mỗi nhận xét.")
        self.selector = EvidenceSelector(self.cfg["alert_policy"]["content"])

    def thong_ke(self):
        levels = Counter()
        for rid, counts in self.label_counts.items():
            if rid not in self.a8:
                level = score_review(counts)["level"]
                if level:
                    levels[TEN_VAN_DE[level]] += 1
        return {"nhan_xet_chi_co_phan_anh_khia_canh": sum(levels.values()),
                "muc_do_van_de": dict(levels), "su_co_nghiem_trong": len(self.a8),
                "trang_thai_su_co": dict(Counter(x["review_status"] for x in self.a8.values())),
                "tong_nhan_xet_canh_bao": sum(levels.values()) + len(self.a8)}

    def chon_ma(self, review_id="", loai="tat_ca", so_mau_moi_nhom=1, hat_giong=None):
        """Lấy mẫu đều trên các mã duy nhất trong từng nhóm, loại trùng giữa các nhóm."""
        require(loai in {"tat_ca", "a8", "a1_a7"}, 'loai phải là "tat_ca", "a8" hoặc "a1_a7".')
        require(type(so_mau_moi_nhom) is int and so_mau_moi_nhom > 0,
                "Số mẫu mỗi nhóm phải là số nguyên dương.")
        require(isinstance(review_id, str), "review_id phải là chuỗi.")
        review_id = review_id.strip()
        if review_id:
            require(review_id in self.master, "Không tìm thấy mã nhận xét trong system: " + review_id)
            return [review_id]
        nhom = {k: set() for k in THU_TU_NHOM
                if loai == "tat_ca" or (loai == "a8") == (k[0] == "a8_incident")}
        for rid, counts in self.label_counts.items():
            if rid in self.a8:
                key = ("a8_incident", self.a8[rid]["review_status"])
            else:
                level = score_review(counts)["level"]
                key = ("review_problem", level)
            if key in nhom:
                nhom[key].add(rid)
        rng = random.Random(hat_giong)  # None: mỗi lần chọn mới; số nguyên: lặp lại mẫu khi kiểm tra.
        chosen, used = [], set()
        for key, ids in nhom.items():
            candidates = sorted(ids - used)
            sample = rng.sample(candidates, min(so_mau_moi_nhom, len(candidates)))
            chosen.extend(sample)
            used.update(sample)
        return chosen

    def nap_nhan_xet(self, ids):
        """Lấy đủ khía cạnh, A8 và mức độ cho mỗi mã đã chọn, kể cả POS/NEU."""
        require(len(ids) == len(set(ids)), "Danh sách nhận xét được chọn có mã trùng.")
        require(set(ids) <= self.master.keys(), "Danh sách có mã nằm ngoài system.")
        selected = set(ids)
        by_aspect, alerts, incidents = defaultdict(dict), defaultdict(dict), {}
        for source in csv_rows(self.inputs["paths"]["aspects"]):
            if source["review_id"] in selected:
                rid, aspect = source["review_id"], source["aspect"]
                require(aspect not in by_aspect[rid], "Trùng cặp khía cạnh của nhận xét: " + rid)
                by_aspect[rid][aspect] = source
        for row in csv_rows(self.paths["data"]):
            rid = row["review_id"]
            if rid not in selected:
                continue
            if row["record_type"] == "a8_incident":
                require(rid not in incidents, "Trùng sự cố trong kết quả: " + rid)
                incidents[rid] = row
            else:
                require(row["aspect"] not in alerts[rid], "Trùng phản ánh khía cạnh: " + rid)
                alerts[rid][row["aspect"]] = row
        cards = []
        for rid in ids:
            require(set(by_aspect[rid]) == set(ASPECT_NAMES), "Nhận xét thiếu một trong bảy khía cạnh: " + rid)
            source_a8 = self.a8.get(rid)
            require(bool(source_a8) == (rid in incidents), "Sự cố không khớp giữa system và tầng 1: " + rid)
            if source_a8:
                require(source_a8["A8_label"] == "1" and source_a8["review_status"] in TEN_TRANG_THAI,
                        "Nhãn hoặc trạng thái sự cố không hợp lệ: " + rid)
                for key in ("incident_types", "review_status", "annotation_note"):
                    require(source_a8[key] == incidents[rid][key], "Thông tin sự cố bị lệch: " + key)
            aspects = []
            for code, name in ASPECT_NAMES.items():
                source = by_aspect[rid][code]
                label = source["label"]  # Ưu tiên nhãn tay đúng như system; không dùng model_label thay thế.
                require(label in {"NONE", *TEN_NHAN}, "Nhãn cảm xúc không hợp lệ.")
                alert = alerts[rid].get(code)
                require((label in {"NEG", "MIX"}) == bool(alert), "Thiếu/thừa phản ánh NEG/MIX: " + rid + "/" + code)
                if label == "NONE":
                    continue
                if alert:
                    require(alert["predicted_label"] == label, "Nhãn phản ánh khác system.")
                    require(alert["severity"] in TEN_MUC_DO, "Mức độ không hợp lệ.")
                reasons = list(json.loads(alert["review_reasons_json"])) if alert else []
                if not alert:
                    reasons.extend(json.loads(source["evidence_quality_flags_json"]))
                    if dung(source["evidence_needs_review"]):
                        reasons.append("evidence_needs_review")
                    if not dung(source["label_verified"]):
                        reasons.append("sentiment_label_unverified")
                aspects.append({"code": code, "name": name, "label": label,
                                "severity": alert["severity"] if alert else None,
                                "reasons": list(dict.fromkeys(reasons)), "source": source, "alert": alert})
            cards.append({"review_id": rid, "master": self.master[rid], "aspects": aspects,
                          "incident": source_a8, "incident_alert": incidents.get(rid),
                          "has_alert": bool(alerts[rid] or source_a8)})
        for card in cards:
            text = card["master"]["review_text_nlp"]
            card["problem"] = score_review([x["label"] for x in card["aspects"]], bool(card["incident"]))
            for aspect in card["aspects"]:
                source = aspect["source"]
                if source['evidence_text']:
                    aspect['proposal'] = giu_bang_chung(text,source['evidence_text'],source['evidence_spans_json'],
                                                       source['label_source']=='manual' and dung(source['label_verified']))
                else:
                    aspect['proposal'] = {'kind':'context','spans':self.selector.context(text,aspect['code']), 'complete':False}
            if card["incident"]:
                incident = card["incident"]
                if incident['A8_evidence']:
                    card['incident_proposal'] = giu_bang_chung(text,incident['A8_evidence'],incident['A8_evidence_spans_json'],
                                                              incident['review_status']=='DA_DUYET')
                else:
                    card['incident_proposal'] = {'kind':'context','spans':self.selector.context(text), 'complete':False}
        order = {"nang": 1, "vua": 2, "nhe": 3, None: 4}
        return sorted(cards, key=lambda c: 0 if c["incident"] else order[c["problem"]["level"]])


def gom_bang_chung(card):
    """Gộp đoạn đề xuất dùng chung; giữ nguyên liên kết đến mọi khía cạnh/sự cố."""
    text = card["master"]["review_text_nlp"]
    sources = [(a["code"] + " — " + a["name"], a["proposal"]) for a in card["aspects"]]
    if card["incident"]:
        sources.insert(0, ("Sự cố nghiêm trọng", card["incident_proposal"]))
    groups = {}
    for purpose, proposal in sources:
        for span in proposal["spans"]:
            start, end = span["start"], span["end"]
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                    and text[start:end] == span["text"], "Đoạn đề xuất không khớp nhận xét gốc.")
            key = (start, end, span["text"], proposal['kind'])
            item = groups.setdefault(key, {"text": span["text"], "purposes": [], "kind": proposal["kind"]})
            label = purpose + (" (đối chiếu)" if proposal["kind"] == "context" else "")
            if label not in item["purposes"]:
                item["purposes"].append(label)
            if proposal["kind"] == "evidence":
                item["kind"] = "evidence"
        for fragment in proposal.get('unlocated_fragments',[]):
            require(fragment in text, 'Bằng chứng chưa định vị không nguyên văn.')
            item = groups.setdefault(('unlocated',fragment,proposal['kind']),
                                     {'text':fragment,'purposes':[],'kind':proposal['kind']})
            label = purpose+' (chưa xác định vị trí duy nhất)'
            if label not in item['purposes']:item['purposes'].append(label)
    return list(groups.values()), []


STYLE = """
<style>
.foody-t1{font:14px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#18283c;max-width:1040px;margin:12px 0;text-align:left}
.foody-t1 *{box-sizing:border-box}.foody-t1 .t1-intro{color:#536478;margin:0 0 16px}
.foody-t1 .t1-card{border:1px solid #d6e0eb;border-radius:14px;background:#fff;margin:0 0 22px;overflow:hidden}
.foody-t1 .t1-head{padding:20px 24px 15px;background:#f6f9fc;border-bottom:1px solid #e0e7ef}
.foody-t1 .t1-incident .t1-head{background:#fff7f4;border-top:4px solid #b83d2d}
.foody-t1 .t1-title{font-size:16px;letter-spacing:.025em;font-weight:750;color:#213c60;margin:0 0 10px}
.foody-t1 .t1-incident .t1-title{color:#952c22}.foody-t1 .t1-name{font-size:17px;font-weight:650;margin:12px 0 8px}
.foody-t1 .t1-meta{display:flex;gap:12px 36px;flex-wrap:wrap;color:#526278;font-size:13px}
.foody-t1 .t1-incident-type{margin:13px 0 0;color:#6e3027}.foody-t1 .t1-incident-type strong{font-size:12px;letter-spacing:.04em}
.foody-t1 .t1-badge{display:inline-block;padding:4px 10px;border-radius:6px;font-size:12px;font-weight:700;background:#eef2f6;color:#46556b;white-space:nowrap}
.foody-t1 .t1-confirmed{background:#def1e8;color:#17623f}.foody-t1 .t1-pending{background:#fff0cf;color:#80530d;border:1px solid #ebc671}
.foody-t1 .t1-problem{margin-top:14px;font-weight:650}.foody-t1 .t1-nang{color:#a32e25}.foody-t1 .t1-vua{color:#98620c}.foody-t1 .t1-nhe{color:#365c85}.foody-t1 .t1-main{padding:16px 24px 6px}.foody-t1 .t1-section{font-size:14px;font-weight:700;margin:0 0 10px}
.foody-t1 table{border-collapse:collapse!important;width:100%;margin:0 0 12px!important;font-size:13px;text-align:left!important;table-layout:fixed}
.foody-t1 th{background:#f3f6fa;color:#536478;font-weight:650;padding:10px;text-align:left!important;border-bottom:1px solid #dce5ef}
.foody-t1 td{padding:11px 10px;border-bottom:1px solid #e8edf3;vertical-align:top;text-align:left!important;overflow-wrap:anywhere;background:#fff}
.foody-t1 th:first-child{width:62%}.foody-t1 .t1-aspect-code{color:#748399;font-size:11px;margin-right:5px}
.foody-t1 .t1-POS{background:#e4f3ec;color:#17623f}.foody-t1 .t1-NEG{background:#fdeae7;color:#9b3026}
.foody-t1 .t1-MIX{background:#fff0d9;color:#8a530a}.foody-t1 .t1-NEU{background:#e9eff9;color:#34568b}
.foody-t1 .t1-serious{color:#a32e25;font-weight:700}.foody-t1 .t1-light{color:#8a5c11}.foody-t1 .t1-unknown{color:#64748b}
.foody-t1 details{margin:0 24px;border-top:1px solid #e3eaf2}.foody-t1 summary{cursor:pointer;padding:14px 0;color:#244e85;font-weight:650}
.foody-t1 .t1-details-body{padding:0 0 20px}.foody-t1 .t1-details-body h4{font-size:14px;margin:18px 0 8px;color:#203955}
.foody-t1 .t1-original{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fb;border:1px solid #e0e7ef;border-radius:8px;padding:14px;font:inherit;max-height:none;color:#26374b}
.foody-t1 blockquote{margin:6px 0 14px!important;border-left:3px solid #a9bfd8;padding:8px 13px!important;color:#30445c;background:#f7f9fc;white-space:pre-wrap;overflow-wrap:anywhere}
.foody-t1 .t1-evidence-for{font-size:12px;font-weight:650;color:#536478}.foody-t1 .t1-muted{color:#66778b;font-size:12px}
.foody-t1 .t1-reasons{margin:8px 0;padding-left:20px}.foody-t1 .t1-reasons li{margin:8px 0}.foody-t1 .t1-note{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff8e8;padding:12px;border-radius:7px}
@media(max-width:620px){.foody-t1 .t1-head,.foody-t1 .t1-main{padding-left:14px;padding-right:14px}.foody-t1 details{margin:0 14px}.foody-t1 th:first-child{width:55%}.foody-t1 th,.foody-t1 td{padding:9px 5px}.foody-t1 .t1-badge{white-space:normal}.foody-t1 .t1-meta{gap:5px 16px}}
</style>
"""


def tao_html(cards, hien_toan_van=True, ngau_nhien=True):
    """Giao diện chỉ đọc; mọi chữ lấy từ dữ liệu đều được thoát ký tự HTML."""
    h = lambda value: escape(str(value), quote=True)
    mode = "được chọn ngẫu nhiên" if ngau_nhien else "theo mã được chỉ định"
    parts = [STYLE, '<div class="foody-t1">',
             f'<p class="t1-intro">Đang xem <strong>{len(cards)} nhận xét</strong> {mode}. Mỗi nhận xét hiển thị một lần.</p>']
    for card in cards:
        master, incident = card["master"], card["incident"]
        title = "CẢNH BÁO — SỰ CỐ NGHIÊM TRỌNG" if incident else (
            "PHẢN ÁNH THEO KHÍA CẠNH" if card["has_alert"] else "NỘI DUNG NHẬN XÉT")
        parts += ['<article class="t1-card' + (' t1-incident' if incident else '') + '">',
                  '<div class="t1-head">', f'<div class="t1-title">{title}</div>']
        if incident:
            state = incident["review_status"]
            css = "t1-confirmed" if state == "DA_DUYET" else "t1-pending"
            parts.append(f'<span class="t1-badge {css}">{TEN_TRANG_THAI[state]}</span>')
        parts += [f'<div class="t1-name">Nhà hàng/Quán: {h(master["restaurant_name"])}</div>',
                  '<div class="t1-meta">', f'<span>ID nhà hàng: <strong>{h(master["restaurant_id"])}</strong></span>',
                  f'<span>Ngày đăng: <strong>{h(ngay_hien_thi(master["review_date_iso"]))}</strong></span>', '</div>']
        if incident:
            parts.append('<p class="t1-incident-type"><strong>SỰ CỐ VỀ:</strong> '
                         + h("; ".join(ten_su_co(incident["incident_types"]))) + '</p>')
        elif card["problem"]["level"]:
            level = card["problem"]["level"]
            parts.append(f'<div class="t1-problem">Mức độ vấn đề: <span class="t1-{level}">{TEN_VAN_DE[level]}</span></div>')
        parts += ['</div><div class="t1-main">', '<div class="t1-section">Các khía cạnh được đề cập</div>']
        if card["aspects"]:
            parts.append('<table><thead><tr><th>Khía cạnh</th><th>Nhãn cảm xúc</th></tr></thead><tbody>')
            for aspect in card["aspects"]:
                label = aspect["label"]
                parts.append(f'<tr><td><span class="t1-aspect-code">{h(aspect["code"])}</span>{h(aspect["name"])}</td>'
                             f'<td><span class="t1-badge t1-{label}">{TEN_NHAN[label]} ({label})</span></td></tr>')
            parts.append('</tbody></table>')
        else:
            parts.append('<p class="t1-muted">Không có khía cạnh được đề cập trong các nhãn hiện có.</p>')
        parts += ['</div><details class="t1-common-details"><summary>Xem chi tiết</summary><div class="t1-details-body">']
        evidence, _ = gom_bang_chung(card)
        # Cùng phạm vi sử dụng thì chỉ in tên một lần, kể cả MIX có hai vế.
        grouped = {}
        for item in evidence:
            key = (item["kind"], tuple(item["purposes"]))
            grouped.setdefault(key, []).append(item["text"])
        for kind in ("confirmed_evidence", "evidence", "context"):
            group_items = [(key, values) for key, values in grouped.items() if key[0] == kind]
            if not group_items:
                continue
            parts.append('<h4>' + TEN_BANG_CHUNG[kind] + '</h4>')
            if kind == 'context':
                parts.append('<p class="t1-muted">Các đoạn đối chiếu dưới đây chưa đủ để xác nhận bằng chứng cho nhãn tương ứng.</p>')
            for (_, purposes), fragments in group_items:
                parts.append('<div class="t1-evidence-for">' + h('; '.join(purposes)) + '</div>')
                for fragment in fragments:
                    parts.append('<blockquote>' + h(fragment) + '</blockquote>')
        if incident and incident["annotation_note"]:
            parts += ['<h4>Ghi chú sự cố</h4>', '<div class="t1-note">' + h(incident["annotation_note"]) + '</div>']
        if hien_toan_van:
            parts += ['<details class="t1-original-toggle" style="margin:16px 0 0"><summary>Review gốc</summary>',
                      f'<div class="t1-original">{h(master["review_text_nlp"])}</div></details>']
        parts.append('</div></details></article>')
    parts.append('</div>')
    return ''.join(parts)


def tao_van_ban(cards, hien_toan_van=True):
    """Bản đọc bằng chữ giữ cùng nhãn và mức vấn đề với bản HTML."""
    lines = [f"Đang xem {len(cards)} nhận xét."]
    for card in cards:
        master, incident = card["master"], card["incident"]
        lines += ['', 'CẢNH BÁO — SỰ CỐ NGHIÊM TRỌNG' if incident else
                  'PHẢN ÁNH THEO KHÍA CẠNH' if card['has_alert'] else 'NỘI DUNG NHẬN XÉT']
        if incident:
            lines.append(TEN_TRANG_THAI[incident['review_status']])
        lines += ['Nhà hàng/Quán: ' + master['restaurant_name'], 'ID nhà hàng: ' + master['restaurant_id'],
                  'Ngày đăng: ' + ngay_hien_thi(master['review_date_iso'])]
        if incident:
            lines.append('SỰ CỐ VỀ: ' + '; '.join(ten_su_co(incident['incident_types'])))
        elif card['problem']['level']:
            lines.append('Mức độ vấn đề: ' + TEN_VAN_DE[card['problem']['level']])
        for aspect in card['aspects']:
            lines.append(f'{aspect["code"]} — {aspect["name"]}: {TEN_NHAN[aspect["label"]]} ({aspect["label"]})')
        evidence, _ = gom_bang_chung(card)
        for item in evidence:
            title = TEN_BANG_CHUNG[item['kind']]
            lines.append(title + ' — ' + '; '.join(item['purposes']) + ': ' + item['text'])
        if incident and incident['annotation_note']:
            lines.append('Ghi chú sự cố: ' + incident['annotation_note'])
        if hien_toan_van:
            lines.append('Review gốc: ' + master['review_text_nlp'])
    return '\n'.join(lines)


def xem_canh_bao(review_id="", loai="tat_ca", so_mau_moi_nhom=1, hien_toan_van=True,
                 project_root=ROOT, config=None, hat_giong=None, che_do="tu_dong"):
    """Chọn mẫu và hiển thị. loai chỉ lọc việc chọn mẫu; mỗi mẫu luôn có đủ A1–A7/A8 hiện có."""
    require(che_do in {"tu_dong", "html", "van_ban"}, "Chế độ hiển thị không hợp lệ.")
    reader = BoDocTang1(project_root=project_root, config=config)
    ids = reader.chon_ma(review_id, loai, so_mau_moi_nhom, hat_giong)
    if not ids:
        print("Không có cảnh báo phù hợp trong kết quả hiện tại.")
        return
    cards = reader.nap_nhan_xet(ids)
    if che_do != "van_ban":
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
            if che_do == "html" or get_ipython() is not None:
                display(HTML(tao_html(cards, hien_toan_van, ngau_nhien=not bool(review_id.strip()))))
                return
        except ImportError:
            pass
    print(tao_van_ban(cards, hien_toan_van))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--review-id", default="")
    parser.add_argument("--loai", choices=["tat_ca", "a8", "a1_a7"], default="tat_ca")
    parser.add_argument("--so-mau-moi-nhom", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        reader = BoDocTang1(project_root=args.project_root, kiem_tra_day_du=True)
        print("Kết quả system và tầng 1 hợp lệ:", json.dumps(reader.thong_ke(), ensure_ascii=False))
        return 0
    xem_canh_bao(project_root=args.project_root, review_id=args.review_id, loai=args.loai,
                 so_mau_moi_nhom=args.so_mau_moi_nhom, che_do="van_ban")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError, OSError) as exc:
        print("DỪNG:", exc, file=sys.stderr)
        raise SystemExit(2)
