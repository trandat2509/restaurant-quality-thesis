"""Tầng 1: phản ánh A1–A7 và A8 đã duyệt, dùng system chung, không chạy mô hình.

Chạy từ gốc dự án: python -m src.alerts.build_content_alerts
Luật đọc tại configs/system/foody_v01.json; không phụ thuộc bộ alerts cũ.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.phobert_system_common import load_settings, location
from src.pipeline.prepare_system_data import csv_rows, load_system_inputs, verify_system, system_paths
from src.pipeline.prepare_system_data import LAYOUT_VERSION, alert_output_dir

VERSION = "foody_content_alerts_unified_v01"
ASPECT_NAMES = {"A1": "Thái độ phục vụ", "A2": "Thời gian phục vụ", "A3": "Chất lượng món ăn",
                "A4": "Không gian và cơ sở vật chất", "A5": "Vệ sinh", "A6": "Giá cả",
                "A7": "Tiếp nhận và xử lý phản hồi"}
LEVEL_NAMES = {"light": "NHẸ", "serious": "NGHIÊM TRỌNG", "unknown": "CHƯA XÁC ĐỊNH"}
EVIDENCE_COPY = ["evidence_text", "evidence_spans_json", "evidence_source", "evidence_status",
                 "evidence_verified", "evidence_needs_review", "evidence_display_mode",
                 "evidence_quality_flags_json", "evidence_model_version", "evidence_text_sha256"]
META = ["review_id", "restaurant_id", "restaurant_name", "review_date_iso", "city_std", "text_group_id"]
COLUMNS = ["alert_id", "record_type", *META, "aspect", "aspect_name", "predicted_label",
           "sentiment_source", "label_verified", "model_label", "model_confidence", "model_margin",
           "model_version", "severity", "severity_title", "severity_source", "severity_verified",
           "severity_rule_ids_json", "severity_evidence_text", "severity_evidence_spans_json",
           "severity_rule_matches_json", "severity_reason", *EVIDENCE_COPY,
           "evidence_start", "evidence_end", "A8_label", "a8_confirmed", "A8_alert_title",
           "incident_types", "a8_label_source", "a8_evidence_source", "annotation_note",
           "model_A8_label", "p_A8", "review_has_a8_alert", "A8_review_status", "review_status",
           "needs_manual_review", "review_reasons_json", "priority", "priority_title",
           "decision_source", "has_manual_label", "used_for_training", "excluded_vlsp_overlap",
           "rule_version", "review_text_nlp"]
BLOCKING_FLAGS = {"missing_evidence", "very_short_fragment", "fragment_inside_word",
                  "ambiguous_manual_position", "manual_annotation_marked_uncertain",
                  "mixed_polarity_not_separated"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def flag(value):
    require(str(value).lower() in {"true", "false", "1", "0"}, "Cờ boolean không hợp lệ: " + str(value))
    return str(value).lower() in {"true", "1"}


def truth(value):
    return str(bool(value)).lower()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def spans_from_json(value, text):
    spans = json.loads(value)
    require(isinstance(spans, list), "Danh sách đoạn trích không hợp lệ.")
    for s in spans:
        a, b = s.get("start"), s.get("end")
        require(type(a) is int and type(b) is int and 0 <= a < b <= len(text), "Vị trí đoạn trích không hợp lệ.")
        require(s.get("text") == text[a:b], "Đoạn trích không khớp nguyên văn.")
    return spans


def contexts(text):
    """Tách câu/mệnh đề, giữ nguyên vị trí Unicode; không cắt dấu chấm thập phân."""
    edges = [0] + [m.end() for m in re.finditer(r"(?<!\d)[.!?;]+(?!\d)|\n+", text)] + [len(text)]
    result = []
    for left, right in zip(edges, edges[1:]):
        piece = text[left:right]
        a = left + len(piece) - len(piece.lstrip())
        b = left + len(piece.rstrip())
        if a < b:
            result.append((a, b))
    return result


def overlap(a, b, c, d):
    return a < d and b > c


NUMBER = r"\d+(?:[.,]\d+)?|một|hai|ba|bốn|năm|nửa"
DURATION = re.compile(r"(?<!\w)(?P<number>" + NUMBER + r")\s*(?P<unit>phút|tiếng|giờ)(?!\w)(?P<half>\s+rưỡi)?", re.I)
RANGE_DURATION = re.compile(r"(?<!\w)(?P<lo>" + NUMBER + r")\s*(?:-|–|đến|tới)\s*(?P<hi>" + NUMBER + r")\s*(?P<unit>phút|tiếng|giờ)(?!\w)", re.I)


def number(value):
    words = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "nửa": 0.5}
    return words[value.lower()] if value.lower() in words else float(value.replace(",", "."))


def wait_events(piece):
    """Đọc thời lượng sau chờ/đợi; tránh giờ mở cửa, khoảng dưới ngưỡng và thời gian khác ở xa."""
    found = []
    for anchor in re.finditer(r"(?<!\w)(?:chờ|đợi)(?!\w)", piece, re.I):
        tail = piece[anchor.end():anchor.end()+90]
        # Dừng ở mệnh đề mới; không nhặt thời gian mở cửa sau dấu phẩy.
        tail = re.split(r"[,;.!?\n](?!\d)", tail, maxsplit=1)[0]
        first = DURATION.search(tail)
        ranged = RANGE_DURATION.search(tail)
        if ranged and (not first or ranged.start() <= first.start()):
            value = min(number(ranged['lo']), number(ranged['hi']))
            minutes = value * (1 if ranged['unit'].lower() == "phút" else 60)
            begin, end = ranged.span()
        elif first:
            begin, end = first.span()
            prefix = tail[:begin]
            if re.search(r"(?:gần|chưa đầy|chưa đến|không đến|sắp|lúc|đến|tới)\s*$", prefix, re.I):
                continue
            if re.search(r"(?:sáng|trưa|chiều|tối)\b", tail[end:end+12], re.I):
                continue
            value = number(first['number']) + (0.5 if first['half'] else 0)
            minutes = value * (1 if first['unit'].lower() == "phút" else 60)
            following = DURATION.match(tail, end) or DURATION.match(tail, end+len(tail[end:])-len(tail[end:].lstrip()))
            if first['unit'].lower() != "phút" and following and following['unit'].lower() == "phút":
                minutes += number(following['number'])
                end = following.end()
        else:
            continue
        if re.search(r"(?:từ|lúc|đến|tới)\s*$", tail[:begin], re.I):
            continue
        found.append((minutes, anchor.start(), anchor.end()+end))
    return found


# Chỉ là bộ chặn ngữ cảnh; các luật xác định mức độ vẫn lấy từ cấu hình chung.
DENIAL_TERMS = (r"chửi|mắng|lăng mạ|sỉ nhục|xúc phạm|phân biệt đối xử|đe dọa|hăm dọa|"
                r"ôi thiu|thiu|hỏng|mốc|chập điện|rò điện|điện giật|sập trần|"
                r"giòi|gián|sâu|ruồi|dị vật|ngộ độc|đau bụng|tiêu chảy|nôn mửa|"
                r"tính khống|thu khống|chặt chém|lừa đảo|từ chối|cộc cằn|thờ ơ|"
                r"bất lịch sự|quá mặn|nguội|tanh|khó chịu")
DENIAL = re.compile(r"(?<!\w)(?:không|ko|chưa|chẳng)\s+(?:(?:hề|bị|có|từng|phải|ai)\s+){0,4}(?:" + DENIAL_TERMS + r")(?!\w)", re.I)
HYPOTHETICAL = re.compile(r"(?<!\w)(?:nếu|giả sử|lỡ|sợ|lo rằng|lo bị|nghi ngờ|hình như|có vẻ)(?!\w)", re.I)


def meal_context(piece, symptom):
    if re.search(r"sau\s+(?:khi\s+)?(?:ăn|uống)(?!\w)", piece[symptom.end():symptom.end()+80], re.I):
        return True
    for meal in re.finditer(r"(?<!\w)(?:ăn|uống)(?!\w)", piece[:symptom.start()], re.I):
        prefix = piece[max(0, meal.start()-45):meal.start()]
        if re.search(r"(?:quán|món|đồ|thức)\s*$", prefix, re.I):
            continue
        if re.search(r"(?:không|ko|chưa|nếu|sợ)(?:\s+\w+){0,3}\s*$", prefix, re.I):
            continue
        if symptom.start()-meal.end() <= 240:
            return True
    return False


def contamination_in_food(fragment, after=''):
    """Không coi ruồi/gián quanh quán là dị vật nằm trong món chỉ vì cùng câu có từ đồ ăn."""
    if re.search(r'ôi thiu|hết hạn', fragment, re.I):
        return True
    if re.match(r'\s*(?:bay|bò.{0,10}(?:bàn|sàn|nền)|trong quán|quanh quán)', after, re.I):
        return False
    food = r'(?:món|đồ ăn|thức ăn|cơm|bún|mì|tô|bát|dĩa|đĩa|ly|cốc|nước|canh|miến|lẩu)'
    contaminant = r'(?:giòi|gián|sâu|ruồi|dị vật|mảnh nhựa|mảnh kính|mảnh sắt|tóc)'
    ambient = r'quán|nhà vệ sinh|không gian|xung quanh|ngoài|sàn|bàn ghế|bay|bò lên bàn'
    direct = r'(?<!\w)'+food+r'(?!\w)(?P<link>.{0,55}?)(?<!\w)'+contaminant+r'(?!\w)'
    for match in re.finditer(direct, fragment, re.I):
        link = match['link']
        if re.search(r'(?<!\w)(?:có|lẫn|dính|rơi|rớt|nằm|chứa|thấy)(?!\w)', link, re.I) and not re.search(ambient, link, re.I):
            return True
    reverse = r'(?<!\w)'+contaminant+r'(?!\w)(?P<link>.{0,55}?)(?<!\w)'+food+r'(?!\w)'
    for match in re.finditer(reverse, fragment, re.I):
        link = match['link']
        if re.search(r'(?<!\w)(?:trong|rớt vào|rơi vào|lẫn vào)(?!\w)', link, re.I) and not re.search(ambient, link, re.I):
            return True
    return False


class SeverityEngine:
    def __init__(self, policy):
        require(set(policy.get("reflection_labels", [])) == {"NEG", "MIX"}, "Tầng 1 yêu cầu reflection_labels gồm NEG và MIX.")
        require(set(policy.get("severity_labels", [])) == {"light", "serious", "unknown"}, "Thiếu ba mức độ đã chốt.")
        self.policy = policy
        self.rules = policy["aspect_rules"]
        require(len({r["id"] for r in self.rules}) == len(self.rules), "Trùng mã luật mức độ.")
        self.compiled = {}
        for rule in self.rules:
            require(rule["aspect"] in ASPECT_NAMES and rule["severity"] in {"light", "serious"}, "Luật khía cạnh hoặc mức độ sai.")
            self.compiled[rule['id']] = {k: [re.compile(r"(?<!\w)(?:"+p+r")(?!\w)", re.I) for p in rule.get(k, [])]
                                         for k in ["patterns_any", "patterns_all", "exclude_patterns"]}
            require(rule.get("patterns_any") or "min_wait_minutes" in rule, "Luật thiếu điều kiện: " + rule['id'])
        cp = policy["confidence_policy"]
        require(cp.get("affects_complaint_rate") is False, "Điểm mô hình không được lọc mẫu số thống kê.")

    def analyse(self, text, row):
        spans = spans_from_json(row['evidence_spans_json'], text)
        flags = set(json.loads(row['evidence_quality_flags_json']))
        if not spans:
            return "unknown", [], "missing_or_unlocatable_evidence"
        if flags & BLOCKING_FLAGS:
            return "unknown", [], "evidence_requires_context_review"
        matches, blocked = [], False
        for start, end in contexts(text):
            if not any(overlap(start, end, s['start'], s['end']) for s in spans):
                continue
            piece = text[start:end]
            for rule in self.rules:
                if rule['aspect'] != row['aspect']:
                    continue
                compiled = self.compiled[rule['id']]
                exclusions = [m.span() for p in compiled['exclude_patterns'] for m in p.finditer(piece)]
                exclusions += [m.span() for m in DENIAL.finditer(piece)]
                def accepted(a, b):
                    return not any(overlap(a, b, x, y) for x, y in exclusions)
                if any(not any(accepted(*m.span()) for m in p.finditer(piece)) for p in compiled['patterns_all']):
                    continue
                candidates = []
                if 'min_wait_minutes' in rule:
                    candidates = [(a, b) for value, a, b in wait_events(piece) if value >= rule['min_wait_minutes']]
                else:
                    for pattern in compiled['patterns_any']:
                        for match in pattern.finditer(piece):
                            if rule['id'] == 'A5_SERIOUS_HEALTH' and not meal_context(piece, match):
                                blocked = True
                                continue
                            if rule['id'] == 'A5_SERIOUS_CONTAMINATION' and not contamination_in_food(match.group(), piece[match.end():match.end()+25]):
                                blocked = True
                                continue
                            candidates.append(match.span())
                triggers = []
                for a, b in candidates:
                    if not any(overlap(start+a, start+b, s['start'], s['end']) for s in spans):
                        continue
                    if not accepted(a, b):
                        blocked = True
                        continue
                    prefix = piece[max(0, a-55):a]
                    # Không nâng mức độ từ giả định hoặc nghi ngờ đứng ngay trước căn cứ.
                    if HYPOTHETICAL.search(piece[a:b]) or (HYPOTHETICAL.search(prefix) and not re.search(r"[,;:!?]", prefix)):
                        blocked = True
                        continue
                    triggers.append({"start": start+a, "end": start+b, "text": text[start+a:start+b]})
                if triggers:
                    matches.append({"rule_id": rule['id'], "severity": rule['severity'], "start": start,
                                    "end": end, "text": text[start:end], "triggers": triggers})
        if not matches:
            return "unknown", [], "negated_or_ambiguous_context" if blocked else "no_aspect_rule_matched_evidence"
        level = "serious" if any(m['severity'] == 'serious' for m in matches) else "light"
        return level, [m for m in matches if m['severity'] == level], "matched_aspect_rule"


def empty_record(master):
    row = dict.fromkeys(COLUMNS, "")
    row.update({key: master[key] for key in META})
    row.update({key: master[key] for key in ['has_manual_label', 'used_for_training', 'excluded_vlsp_overlap']})
    row.update(rule_version=VERSION, review_text_nlp=master['review_text_nlp'], severity_verified="false",
               severity_rule_ids_json="[]", severity_evidence_spans_json="[]", severity_rule_matches_json="[]")
    return row


def make_aspect(master, source, engine):
    text = master['review_text_nlp']
    require(source['evidence_text_sha256'] == text_sha(text), "Bằng chứng không thuộc văn bản hiện tại.")
    row = empty_record(master)
    level, matches, reason = engine.analyse(text, source)
    contexts_unique = {(m['start'], m['end']): {k: m[k] for k in ['start', 'end', 'text']} for m in matches}
    severity_spans = [contexts_unique[k] for k in sorted(contexts_unique)]
    manual = source['label_source'] == 'manual'
    reasons = list(json.loads(source['evidence_quality_flags_json']))
    if not flag(source['label_verified']):
        reasons.append('sentiment_label_unverified')
    if flag(source['evidence_needs_review']):
        reasons.append('evidence_needs_review')
    if not manual:
        cp = engine.policy['confidence_policy']
        if source.get('model_confidence', '') != '' and float(source['model_confidence']) < cp['review_below']:
            reasons.append('low_sentiment_confidence')
        if source.get('model_margin', '') != '' and float(source['model_margin']) < cp['review_margin_below']:
            reasons.append('small_sentiment_margin')
        if source.get('model_confidence', '') == '' or source.get('model_margin', '') == '':
            reasons.append('sentiment_scores_not_exported')
    if level == 'unknown':
        reasons.append(reason)
    spans = spans_from_json(source['evidence_spans_json'], text)
    a8_status = master['A8_review_status'] if master['A8_label'] == '1' else ''
    row.update(alert_id=f"aspect:{source['review_id']}:{source['aspect']}", record_type='aspect_reflection',
               aspect=source['aspect'], aspect_name=ASPECT_NAMES[source['aspect']], predicted_label=source['label'],
               sentiment_source=source['label_source'], label_verified=source['label_verified'], model_label=source['model_label'],
               model_confidence='' if manual else source['model_confidence'], model_margin='' if manual else source['model_margin'],
               model_version=source['model_version'], severity=level, severity_title=LEVEL_NAMES[level],
               severity_source='aspect_rule' if matches else 'unassigned', severity_reason=reason,
               severity_rule_ids_json=compact(sorted({m['rule_id'] for m in matches})),
               severity_evidence_text=' || '.join(s['text'] for s in severity_spans),
               severity_evidence_spans_json=compact(severity_spans), severity_rule_matches_json=compact(matches),
               **{key: source[key] for key in EVIDENCE_COPY},
               evidence_start=spans[0]['start'] if len(spans) == 1 else '', evidence_end=spans[0]['end'] if len(spans) == 1 else '',
               review_has_a8_alert=truth(master['A8_label'] == '1'), A8_review_status=a8_status,
               review_status='CAN_XEM_XET' if reasons else 'CHUA_DUYET_MUC_DO',
               needs_manual_review=truth(reasons), review_reasons_json=compact(sorted(set(reasons))),
               priority='high' if level == 'serious' else 'normal',
               priority_title='ƯU TIÊN XEM' if level == 'serious' else 'THEO DÕI',
               decision_source='manual_sentiment_and_rule' if manual and matches else 'manual_sentiment' if manual else 'model_sentiment_and_rule' if matches else 'model_sentiment')
    return row


def make_a8(master, source):
    require(source['A8_label'] == '1' and source['review_status'] in {'DA_DUYET', 'CAN_XEM_XET'}, "Trạng thái A8 không hợp lệ.")
    confirmed = source['review_status'] == 'DA_DUYET'
    spans = spans_from_json(source['A8_evidence_spans_json'], master['review_text_nlp'])
    row = empty_record(master)
    row.update(alert_id='a8:'+source['review_id'], record_type='a8_incident', aspect='A8', aspect_name='Sự cố bất thường',
               label_verified=truth(confirmed), severity='serious', severity_title='SỰ CỐ A8 — ƯU TIÊN XEM',
               severity_source='reviewed_a8_label', severity_verified=truth(confirmed), severity_reason='preserved_a8_decision',
               evidence_text=source['A8_evidence'], evidence_spans_json=source['A8_evidence_spans_json'],
               evidence_source=source['A8_evidence_source'], evidence_status='a8_reviewed' if confirmed else 'a8_needs_review',
               evidence_verified=truth(confirmed and bool(spans)), evidence_needs_review=truth(not confirmed or not spans),
               evidence_display_mode='quoted_spans_with_full_review' if spans else 'full_review',
               evidence_quality_flags_json=compact([] if spans else ['missing_evidence']),
               evidence_text_sha256=text_sha(master['review_text_nlp']),
               evidence_start=spans[0]['start'] if len(spans) == 1 else '', evidence_end=spans[0]['end'] if len(spans) == 1 else '',
               A8_label='1', a8_confirmed=truth(confirmed), A8_alert_title=source['A8_alert_title'],
               incident_types=source['incident_types'], a8_label_source=source['label_source'],
               a8_evidence_source=source['A8_evidence_source'], annotation_note=source['annotation_note'],
               model_A8_label=source['model_A8_label'], p_A8=source['p_A8'], model_version=source['model_version'],
               review_has_a8_alert='true', A8_review_status=source['review_status'], review_status=source['review_status'],
               needs_manual_review=truth(not confirmed or not spans),
               review_reasons_json=compact((['a8_requires_context_review'] if not confirmed else []) + ([] if spans else ['missing_evidence'])),
               priority='urgent', priority_title='ƯU TIÊN SỰ CỐ A8', decision_source='reviewed_a8_list')
    return row


def audit_record(row, master, allowed_rules):
    require(set(row) == set(COLUMNS) and None not in row and all(v is not None for v in row.values()), "Cột đầu ra thiếu/thừa.")
    text = master['review_text_nlp']
    require(all(row[k] == master[k] for k in META) and row['review_text_nlp'] == text, "Liên kết nhận xét/quán/ngày sai.")
    spans = spans_from_json(row['evidence_spans_json'], text)
    spans_from_json(row['severity_evidence_spans_json'], text)
    if row['record_type'] == 'aspect_reflection':
        require(row['evidence_text'] == ' || '.join(s['text'] for s in spans) or row['evidence_source'] == 'manual', "Chuỗi bằng chứng máy sai.")
        if row['evidence_source'] == 'manual':
            require(all(p.strip() in text for p in row['evidence_text'].split('||') if p.strip()), "Bằng chứng tay không nguyên văn.")
        require(row['severity'] in LEVEL_NAMES and row['severity_verified'] == 'false', "Mức độ luật bị coi là nhãn đã duyệt.")
        ids = json.loads(row['severity_rule_ids_json'])
        require(all(i in allowed_rules and allowed_rules[i]['aspect'] == row['aspect'] and allowed_rules[i]['severity'] == row['severity'] for i in ids), "Luật không thuộc khía cạnh/mức độ.")
        require(bool(ids) == (row['severity'] != 'unknown'), "Mức độ không có căn cứ luật.")
        for match in json.loads(row['severity_rule_matches_json']):
            require(match['rule_id'] in ids and text[match['start']:match['end']] == match['text'], "Ngữ cảnh luật không nguyên văn.")
            for trigger in match['triggers']:
                require(text[trigger['start']:trigger['end']] == trigger['text'], "Từ/cụm khớp luật không nguyên văn.")
                require(any(overlap(trigger['start'], trigger['end'], s['start'], s['end']) for s in spans), "Luật không liên quan đoạn bằng chứng.")
    else:
        require(row['record_type'] == 'a8_incident' and row['A8_label'] == '1', "Loại cảnh báo không hợp lệ.")
    if len(spans) != 1:
        require(row['evidence_start'] == row['evidence_end'] == '', "Đoạn ghép không được gắn một vị trí duy nhất.")
    else:
        require(str(row['evidence_start']) == str(spans[0]['start']) and str(row['evidence_end']) == str(spans[0]['end']), "Cột vị trí không khớp.")


def output_paths(cfg):
    directory = alert_output_dir(cfg, 1)
    return {'data': directory/'content_alerts.csv',
            'report': directory/'kiem_tra_tang_1.md',
            'manifest': directory/'content_run_manifest.json'}


def signature(cfg, system_manifest):
    paths = system_paths(cfg)
    require(sha(cfg['_config']) == cfg['_config_sha256'], 'Cấu hình đã đổi trong lúc chạy; chạy lại để đọc bản mới.')
    return {'version': VERSION, 'config_sha256': sha(cfg['_config']), 'program_sha256': sha(__file__),
            'system_manifest_sha256': sha(paths['manifest']),
            'system_outputs_sha256': system_manifest['output_sha256'],
            'master_sha256': sha(paths['master']),
            'system_reader_sha256': sha(Path(__file__).parents[1]/'pipeline/prepare_system_data.py'),
            'update_reader_sha256': sha(Path(__file__).parents[1]/'pipeline/update_system_data.py'),
            'common_reader_sha256': sha(Path(__file__).parents[1]/'modeling/phobert_system_common.py')}


def validated_inputs(cfg, use_system=False):
    require(cfg.get('system_schema_version') == 2, 'Cần cài SYSTEM_CHUNG.zip và chạy cập nhật system trước.')
    require(cfg['alert_policy']['a8']['use_legacy_incident_rules'] is False, 'Không sử dụng luật A8 cũ.')
    manifest = verify_system(cfg, check_sources=not use_system)
    inputs = load_system_inputs(cfg['_root'], view='operational', config=cfg['_config'])
    require(inputs['manifest'] == manifest, 'System vừa được cập nhật trong lúc đọc; chạy lại khi đầu vào ổn định.')
    return inputs, signature(cfg, manifest)


def audit_file(path, cfg, inputs, engine):
    """Đối chiếu từng dòng đầu ra với đúng nhãn/bằng chứng nguồn, không cần giữ toàn CSV trong RAM."""
    counts, aspects, sources, reviews = Counter(), Counter(), Counter(), set()
    levels, reasons, statuses = Counter(), Counter(), Counter()
    allowed_rules = {r['id']: r for r in engine.rules}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == COLUMNS, 'Đầu ra không đúng cấu trúc tầng 1 hiện tại.')
        for source in csv_rows(inputs['paths']['aspects']):
            counts['input_aspect_pairs'] += 1
            if source['label'] not in {'NEG', 'MIX'}:
                continue
            row = next(reader, None)
            require(row is not None and row['record_type'] == 'aspect_reflection' and row['review_id'] == source['review_id'] and row['aspect'] == source['aspect'], 'Thiếu/trùng/sai thứ tự phản ánh NEG/MIX.')
            require(row['predicted_label'] == source['label'] and row['sentiment_source'] == source['label_source'], 'Cảnh báo dùng sai nguồn nhãn.')
            require(all(row[k] == source[k] for k in EVIDENCE_COPY), 'Cảnh báo đã thay đổi bằng chứng nguồn.')
            require(row['model_label'] == source['model_label'] and row['model_version'] == source['model_version'], 'Thông tin mô hình đã đổi.')
            require(source['label_source'] != 'manual' or row['model_confidence'] == row['model_margin'] == '', 'Điểm máy bị gán cho nhãn tay.')
            require(row['alert_id'] == f"aspect:{source['review_id']}:{source['aspect']}", 'Mã phản ánh sai.')
            audit_record(row, inputs['master'][row['review_id']], allowed_rules)
            counts['aspect_reflections'] += 1
            counts['needs_manual_review'] += flag(row['needs_manual_review'])
            aspects[row['aspect']] += 1
            sources[row['sentiment_source']] += 1
            levels[row['severity']] += 1
            reasons.update(json.loads(row['review_reasons_json']))
            reviews.add(row['review_id'])
        for source in inputs['a8']:
            row = next(reader, None)
            require(row is not None and row['record_type'] == 'a8_incident' and row['alert_id'] == 'a8:'+source['review_id'] and row['review_id'] == source['review_id'], 'Thiếu/thừa A8 đã chốt.')
            require(row['review_status'] == source['review_status'] and row['A8_alert_title'] == source['A8_alert_title'], 'Đã thay đổi trạng thái A8.')
            require(row['a8_confirmed'] == truth(source['review_status'] == 'DA_DUYET'), 'A8 cần xem xét bị tự xác nhận.')
            require(row['evidence_text'] == source['A8_evidence'] and row['evidence_spans_json'] == source['A8_evidence_spans_json'], 'Bằng chứng A8 đã đổi.')
            audit_record(row, inputs['master'][row['review_id']], allowed_rules)
            counts['a8_alerts'] += 1
            counts['needs_manual_review'] += flag(row['needs_manual_review'])
            statuses[row['review_status']] += 1
            reviews.add(row['review_id'])
        require(next(reader, None) is None, 'Đầu ra còn dòng ngoài phạm vi A1–A7 và A8 đã chốt.')
    require(counts['input_aspect_pairs'] == inputs['manifest']['summary']['aspect_pairs'], 'Chưa đọc đủ các cặp A1–A7.')
    require(counts['a8_alerts'] == inputs['manifest']['summary']['a8_alerts'], 'Số A8 không khớp system.')
    return {**dict(counts), 'total_alert_rows': counts['aspect_reflections']+counts['a8_alerts'],
            'unique_reviews_with_alert': len(reviews), 'by_aspect': dict(sorted(aspects.items())),
            'aspect_sentiment_sources': dict(sources), 'aspect_severity_counts': dict(levels),
            'a8_by_status': dict(statuses), 'review_reason_counts': dict(sorted(reasons.items())),
            'a8_explicit_negatives_excluded': len(inputs['manifest']['a8_known_negative_review_ids']),
            'a8_new_detections_added': 0, 'tier2_built': False, 'model_retrained': False,
            'independent_severity_accuracy_measured': False}


def report_text(summary):
    lines = ['# Kết quả cảnh báo tầng 1', '',
             f"- Phản ánh A1–A7 (NEG/MIX): **{summary['aspect_reflections']:,}**.",
             f"- A8 đã chốt: **{summary['a8_alerts']:,}**; trạng thái: {compact(summary['a8_by_status'])}.",
             f"- Tổng số dòng cảnh báo: **{summary['total_alert_rows']:,}**.",
             f"- Số nhận xét có ít nhất một cảnh báo: **{summary['unique_reviews_with_alert']:,}**.",
             f"- Nguồn nhãn A1–A7: {compact(summary['aspect_sentiment_sources'])}.",
             f"- Mức độ A1–A7 theo luật: {compact(summary['aspect_severity_counts'])}.", '',
             '## Cách đọc kết quả', '',
             '- Một nhận xét có thể có nhiều dòng A1–A7 và thêm một dòng A8; không cộng các dòng thành số nhận xét.',
             '- Mức độ A1–A7 là gợi ý từ luật, severity_verified=false; chưa có đánh giá độ chính xác độc lập.',
             '- evidence_text/evidence_spans_json giữ bằng chứng nguồn. severity_evidence_text là ngữ cảnh luật, lưu riêng.',
             '- Đoạn thiếu, cắt từ, quá ngắn, vị trí mơ hồ hoặc MIX máy chưa tách khen/chê: chưa tự xếp mức độ.',
             '- unknown là chưa xác định mức độ, không có nghĩa bình thường hoặc đã loại cảnh báo.',
             '- A8 DA_DUYET/CAN_XEM_XET giữ quyết định gán nhãn; chưa phải trạng thái giải quyết công việc của quản lý.',
             '- Không thêm dự đoán A8 ngoài danh sách system. Nhãn 0 đã duyệt bị loại; phần chưa duyệt không tự đổi thành 0.',
             '- Chưa tạo tầng 2, chưa xây giao diện, không huấn luyện hoặc chạy lại mô hình.', '',
             '## Các lý do cần xem xét của A1–A7', '', '| Lý do | Số dòng |', '|---|---:|']
    labels = {'sentiment_label_unverified':'Nhãn cảm xúc chưa được người đọc xác nhận', 'evidence_needs_review':'Bằng chứng cần kiểm tra',
              'low_sentiment_confidence':'Điểm dự đoán thấp', 'small_sentiment_margin':'Hai nhãn có điểm gần nhau',
              'sentiment_scores_not_exported':'Đầu ra nguồn chưa lưu đủ điểm dự đoán',
              'missing_or_unlocatable_evidence':'Thiếu bằng chứng hoặc chưa xác định được vị trí',
              'evidence_requires_context_review':'Bằng chứng chưa đủ điều kiện xếp mức độ',
              'no_aspect_rule_matched_evidence':'Chưa khớp luật mức độ của khía cạnh',
              'negated_or_ambiguous_context':'Ngữ cảnh phủ định hoặc mơ hồ', 'missing_evidence':'Thiếu bằng chứng',
              'very_short_fragment':'Đoạn trích quá ngắn', 'fragment_inside_word':'Đoạn trích cắt giữa từ',
              'ambiguous_manual_position':'Bằng chứng tay có nhiều vị trí giống nhau',
              'manual_annotation_marked_uncertain':'Gán tay có cờ chưa chắc chắn',
              'mixed_polarity_not_separated':'MIX máy chưa tách đoạn khen và chê'}
    lines += [f"| {labels.get(k,k)} | {v:,} |" for k,v in summary['review_reason_counts'].items()]
    lines += ['', 'Một dòng có thể có nhiều lý do; không cộng cột này thành tổng cảnh báo.', '']
    return '\n'.join(lines)


@contextmanager
def output_lock(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    lock = data_dir/'.tang_1_dang_chay.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError('Đang có lần tạo tầng 1 khác hoặc lần trước bị đóng đột ngột. Chỉ xóa '+str(lock)+' sau khi chắc chắn không còn chương trình tầng 1 đang chạy.') from None
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(compact({'pid':os.getpid(), 'created_at_utc':datetime.now(timezone.utc).isoformat()}))
        yield
    finally:
        lock.unlink(missing_ok=True)


def commit_outputs(staging, paths):
    """Thay ba tệp theo thứ tự, manifest cuối; khôi phục bản trước nếu có lỗi giữa chừng."""
    rollback = staging/'rollback'
    rollback.mkdir()
    existed, committed = {}, []
    for key, target in paths.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        existed[key] = target.exists()
        if existed[key]:
            shutil.copy2(target, rollback/key)
    try:
        for key, target in paths.items():
            os.replace(staging/key, target)
            committed.append(key)
    except BaseException:
        for key in reversed(committed):
            if existed[key]:
                os.replace(rollback/key, paths[key])
            else:
                paths[key].unlink(missing_ok=True)
        raise


def verify_outputs(cfg, inputs, identity):
    paths = output_paths(cfg)
    require(paths['manifest'].is_file(), 'Chưa có kết quả tầng 1 hiện tại; chạy lệnh tạo cảnh báo trước.')
    manifest = read_json(paths['manifest'])
    require(manifest.get('status') == 'complete' and manifest.get('version') == VERSION, 'Đầu ra tầng 1 thuộc bản cũ hoặc chưa hoàn thành; chạy lại lệnh tạo.')
    require(manifest['signature'] == identity, 'Đầu vào hoặc luật đã đổi; chạy lại lệnh tạo tầng 1 để cập nhật.')
    for key in ['data', 'report']:
        require(paths[key].is_file() and sha(paths[key]) == manifest['output_sha256'][key], 'Đầu ra thiếu hoặc đã bị sửa: '+str(paths[key]))
    actual = audit_file(paths['data'], cfg, inputs, SeverityEngine(cfg['alert_policy']['content']))
    require(actual == manifest['summary'], 'Số lượng và nội dung đầu ra không khớp báo cáo.')
    return manifest


def run(cfg, verify_only=False, check_only=False, overwrite=False, use_system=False, inputs_override=None):
    if inputs_override is None:
        inputs, identity = validated_inputs(cfg, use_system)
    else:
        inputs, identity = inputs_override, signature(cfg, inputs_override['manifest'])
    cfg = dict(cfg, _system_snapshot=inputs['paths']['directory'])
    identity = signature(cfg, inputs['manifest'])
    engine = SeverityEngine(cfg['alert_policy']['content'])
    paths = output_paths(cfg)
    if check_only:
        print('Đầu vào tầng 1 hợp lệ:', compact(inputs['manifest']['summary']), flush=True)
        return inputs['manifest']
    with output_lock(paths['data'].parent):
        if verify_only:
            result = verify_outputs(cfg, inputs, identity)
            print('Tầng 1 hợp lệ:', compact(result['summary']), flush=True)
            return result
        if not overwrite and paths['manifest'].is_file():
            previous = read_json(paths['manifest'])
            if previous.get('signature') == identity:
                result = verify_outputs(cfg, inputs, identity)
                print('Tầng 1 đã cập nhật, nguồn không đổi:', compact(result['summary']), flush=True)
                return result
        staging = Path(tempfile.mkdtemp(prefix='tang1_', dir=paths['data'].parent.parent))
        try:
            print('Đang tạo phản ánh A1–A7 từ system chung...', flush=True)
            written = 0
            with (staging/'data').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='raise')
                writer.writeheader()
                for source in inputs['aspects']:
                    if source['label'] not in {'NEG', 'MIX'}:
                        continue
                    writer.writerow(make_aspect(inputs['master'][source['review_id']], source, engine))
                    written += 1
                    if written % 10000 == 0:
                        print(f'Đã tạo {written:,} phản ánh A1–A7...', flush=True)
                for source in inputs['a8']:
                    writer.writerow(make_a8(inputs['master'][source['review_id']], source))
            print('Đang đối chiếu nhãn, bằng chứng và đủ danh sách A8...', flush=True)
            summary = audit_file(staging/'data', cfg, inputs, engine)
            (staging/'report').write_text(report_text(summary), encoding='utf-8')
            verify_system(cfg, check_sources=not use_system)
            require(signature(cfg, inputs['manifest']) == identity, 'Nguồn hoặc mã đã đổi trong lúc tạo cảnh báo. Giữ đầu ra trước đó; chạy lại khi nguồn ổn định.')
            manifest = {'version':VERSION, 'status':'complete', 'created_at_utc':datetime.now(timezone.utc).isoformat(),
                        'output_layout_version': LAYOUT_VERSION,
                        'signature':identity, 'summary':summary,
                        'output_paths':{k:str(p.relative_to(cfg['_root'])).replace('\\','/') for k,p in paths.items()},
                        'output_sha256':{k:sha(staging/k) for k in ['data','report']},
                        'severity_is_rule_suggestion':True, 'manual_labels_modified':False,
                        'model_retrained':False, 'tier2_built':False}
            save_json(staging/'manifest', manifest)
            commit_outputs(staging, paths)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    print('Đã hoàn thành tầng 1:', json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Danh sách cảnh báo:', paths['data'], flush=True)
    print('Báo cáo:', paths['report'], flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--config', default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Chủ động tạo lại cùng đầu ra dù nguồn không đổi.')
    parser.add_argument('--use-system', action='store_true', help='Dùng bản system đã kiểm tra; không cần đọc lại nguồn cũ bên ngoài.')
    args = parser.parse_args(argv)
    cfg = load_settings(args.config, args.project_root)
    run(cfg, args.verify_only, args.check_only, args.overwrite, args.use_system)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError, OSError) as exc:
        print('DỪNG:', exc, file=sys.stderr, flush=True)
        if isinstance(exc, PermissionError):
            print('Đóng tệp kết quả đang mở trong Excel hoặc ứng dụng khác rồi chạy lại.', file=sys.stderr, flush=True)
        raise SystemExit(2)
