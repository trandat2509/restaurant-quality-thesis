"""System chung: chuẩn bị nguồn, cập nhật nhãn/bằng chứng và kiểm tra tại foody_v01.

Mặc định chỉ cập nhật các đầu ra vận hành trong thư mục system đang có.
Corpus và tập huấn luyện đã chốt giữ nguyên, không tải hoặc chạy mô hình.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile
import pandas as pd

# Hỗ trợ cả python -m src.pipeline.prepare_system_data và chạy tệp trực tiếp.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.modeling.phobert_system_common import (
    ASPECTS, LABELS, atomic_csv, fingerprint, load_settings, location,
    now, prepared_data, read_csv, reference_config, require_columns,
    sha256_file, validate_reviews,
)

VERSION = "foody_system_unified"
SCHEMA = 2
LAYOUT_VERSION = 3
# Mở rộng cột có tương thích ngược với bản system nền schema 2.
SOURCE_EVIDENCE_SCHEMA = 1
EVIDENCE_FIELDS = ["evidence_text", "evidence_spans_json", "evidence_source", "evidence_status",
                   "evidence_verified", "evidence_needs_review", "evidence_display_mode",
                   "evidence_quality_flags_json", "evidence_model_version", "evidence_text_sha256"]
SOURCE_EVIDENCE_COLUMNS = {
    source:{key:(f"{source}_evidence_version" if key=="evidence_model_version" else f"{source}_{key}")
            for key in EVIDENCE_FIELDS}
    for source in ["model", "manual"]
}
ASPECT_FIELDS = ["review_id", "aspect", "label", "label_source", "label_verified",
                 "manual_label", "manual_uncertain", "model_label", "model_version",
                 *[f"p_{x}" for x in LABELS], "model_confidence", "model_margin",
                 "evidence_text", "evidence_spans_json", "evidence_source", "evidence_status",
                 "evidence_verified", "evidence_needs_review", "evidence_display_mode",
                 "evidence_quality_flags_json", "evidence_model_version", "evidence_text_sha256",
                 "has_manual_label", "used_for_training", "excluded_vlsp_overlap", "review_text_sha256",
                 *SOURCE_EVIDENCE_COLUMNS["model"].values(), *SOURCE_EVIDENCE_COLUMNS["manual"].values()]

A8_FIELDS = ["review_id", "restaurant_id", "restaurant_name", "review_date_iso", "city_std",
             "text_group_id", "A8_label", "review_status", "A8_alert_title", "label_source",
             "incident_types", "A8_evidence", "A8_evidence_spans_json", "A8_evidence_source",
             "annotation_note", "model_A8_label", "p_A8", "model_version", "review_text_nlp"]



def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value) or value in {'.', '..'}:
        raise ValueError('Tên đợt/thư mục kết quả không hợp lệ.')
    return value

def _report_path(cfg, relative):
    root = Path(cfg['_root']).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root / 'reports'):
        raise ValueError('Đầu ra System/cảnh báo phải nằm trong reports/.')
    return target

def system_output_root(cfg):
    return _report_path(cfg, cfg.get('system_report_dir', 'reports/system/foody_v01'))

def alert_output_root(cfg, tier):
    if tier not in {1, 2}:
        raise ValueError('Chỉ có tầng cảnh báo 1 và 2.')
    version = safe_name(cfg.get('system_version', 'foody_v01'))
    return _report_path(cfg, f'reports/alerts/{version}/tang{tier}')

def output_batch_id(cfg):
    if cfg.get('_output_batch_id'):
        return safe_name(cfg['_output_batch_id'])
    directory = Path(system_paths(cfg)['directory']).resolve()
    base = (Path(cfg['_root']) / cfg['system_data_dir']).resolve()
    return 'ban_dau' if directory == base else safe_name(directory.name)

def alert_output_dir(cfg, tier):
    return alert_output_root(cfg, tier) / output_batch_id(cfg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and "review_id" in reader.fieldnames, f"Thiếu review_id: {path}")
        require(len(set(reader.fieldnames)) == len(reader.fieldnames), f"Trùng tên cột: {path}")
        for line, row in enumerate(reader, 2):
            require(None not in row and all(v is not None for v in row.values()),
                    f"CSV thiếu/thừa cột hoặc bị cắt ở dòng {line}: {path}")
            if not any(row.values()):
                continue
            require(row["review_id"].strip() != "", f"Mã rỗng ở dòng {line}: {path}")
            yield row


def keyed(rows, name):
    result = {}
    for r in rows:
        require(r["review_id"] not in result, f"Trùng mã trong {name}: {r['review_id']}")
        result[r["review_id"]] = r
    return result


def flag(value):
    require(str(value).lower() in {"true", "false", "0", "1"}, f"Cờ boolean sai: {value}")
    return str(value).lower() in {"true", "1"}


def resolve(root, value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def choose(root, values, title):
    for value in values:
        p = resolve(root, value)
        if p.is_file():
            return p
    raise FileNotFoundError(f"Thiếu {title}. Các đường dẫn đã kiểm tra:\n" + "\n".join(str(resolve(root, x)) for x in values))


def source_paths(root, cfg):
    return {key: choose(root, values, key) for key, values in cfg["sources"].items()}


def occurrences(text, fragment):
    positions, start = [], 0
    while True:
        start = text.find(fragment, start)
        if start < 0:
            return positions
        positions.append((start, start + len(fragment)))
        start += 1


def manual_evidence(text, label, evidence, uncertain=False):
    fragments = list(dict.fromkeys(s.strip() for s in evidence.split("||") if s.strip()))
    require((label == "NONE") == (not fragments), "Nhãn tay và bằng chứng không nhất quán.")
    spans, flags = [], []
    for fragment in fragments:
        locations = occurrences(text, fragment)
        require(locations, "Bằng chứng gán tay không nằm trong nhận xét.")
        if len(locations) != 1:
            flags.append("ambiguous_manual_position")
        else:
            start, end = locations[0]
            spans.append({"start": start, "end": end, "text": text[start:end]})
    if uncertain:
        flags.append("manual_annotation_marked_uncertain")
    # Không tự chọn vị trí đầu tiên khi một đoạn có nhiều vị trí đúng.
    if "ambiguous_manual_position" in flags:
        spans = []
    status = "not_mentioned_manual" if label == "NONE" else "manual_evidence_needs_review" if flags else "manual_confirmed"
    return {"evidence_text": evidence, "evidence_spans_json": compact(spans), "evidence_source": "manual",
            "evidence_status": status, "evidence_verified": str(not flags).lower(),
            "evidence_needs_review": str(bool(flags)).lower(),
            "evidence_display_mode": "none" if label == "NONE" else "full_review" if flags else "quoted_spans",
            "evidence_quality_flags_json": compact(sorted(set(flags))), "evidence_model_version": "",
            "evidence_text_sha256": text_hash(text)}


def model_evidence(text, row):
    require(row.get("evidence_source", "model") == "model", "Không nhập bằng chứng tay làm bằng chứng máy.")
    require(row["evidence_text_sha256"] == text_hash(text), "Bằng chứng dùng văn bản khác corpus.")
    spans = json.loads(row["evidence_spans_json"])
    require(isinstance(spans, list), "Cột vị trí bằng chứng không phải danh sách.")
    flags = []
    for s in spans:
        start, end = s["start"], s["end"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), "Vị trí bằng chứng không hợp lệ.")
        require(text[start:end] == s["text"], "Đoạn bằng chứng không nguyên văn.")
        if len("".join(s["text"].split())) <= 2:
            flags.append("very_short_fragment")
        if (start > 0 and text[start-1].isalnum() and text[start].isalnum()) or (end < len(text) and text[end-1].isalnum() and text[end].isalnum()):
            flags.append("fragment_inside_word")
    require(row["evidence_text"] == " || ".join(s["text"] for s in spans), "Văn bản và danh sách đoạn trích không khớp.")
    label = row["predicted_label"]
    require(label != "NONE" or not spans, "Nhãn NONE vẫn chứa bằng chứng.")
    if label != "NONE" and not spans:
        flags.append("missing_evidence")
    if label == "MIX" and spans:
        flags.append("mixed_polarity_not_separated")
    return {"evidence_text": row["evidence_text"], "evidence_spans_json": row["evidence_spans_json"],
            "evidence_source": "model", "evidence_status": row["evidence_status"], "evidence_verified": "false",
            "evidence_needs_review": str(label != "NONE").lower(),
            "evidence_display_mode": "none" if label == "NONE" else "full_review" if any(f != "mixed_polarity_not_separated" for f in flags) else "proposed_spans_with_full_review",
            "evidence_quality_flags_json": compact(sorted(set(flags))),
            "evidence_model_version": row["evidence_model_version"], "evidence_text_sha256": row["evidence_text_sha256"]}


def tagged_evidence(evidence, source):
    """Lưu bằng chứng theo nguồn; không ghi đè phần của nguồn còn lại."""
    return {column:evidence.get(key, "") for key,column in SOURCE_EVIDENCE_COLUMNS[source].items()}


def source_evidence(row, source):
    return {key:row.get(column, "") for key,column in SOURCE_EVIDENCE_COLUMNS[source].items()}


def retain_source_evidence(row, text):
    """Bổ sung cột cho dữ liệu cũ trong bộ nhớ, không sửa tệp đã đóng băng.

    Chỉ lấy lại bằng chứng từ đúng nguồn đã được ghi; không lấy bằng chứng tay
    để bù vào phần máy chưa từng được lưu trong bản cũ.
    """
    row.setdefault("review_text_sha256", "")
    require(not row["review_text_sha256"] or row["review_text_sha256"] == text_hash(text), "System liên kết sai văn bản.")
    row["review_text_sha256"] = text_hash(text)
    for source in ["model", "manual"]:
        for column in SOURCE_EVIDENCE_COLUMNS[source].values():row.setdefault(column, "")
        if not row.get(f"{source}_evidence_source") and row.get("evidence_source") == source:
            row.update(tagged_evidence(row, source))
    return row


def select_operational_source(row, text):
    """Nhãn và bằng chứng vận hành cùng nguồn; nhãn tay đã xác nhận được ưu tiên."""
    retain_source_evidence(row, text)
    confirmed = row.get("manual_label") in LABELS and flag(row.get("label_verified") or "false")
    source = "manual" if confirmed else "model"
    label = row["manual_label"] if confirmed else row.get("model_label", "")
    require(label in LABELS, "Chưa có nhãn tay xác nhận hoặc nhãn máy cho cặp nhận xét–khía cạnh.")
    evidence = source_evidence(row, source)
    if not evidence["evidence_source"]:
        # Dữ liệu cũ có thể chưa trích bằng chứng máy; giữ rõ trạng thái thiếu.
        require(source == "model", "Mất bằng chứng nguồn tay đã xác nhận.")
        evidence = {"evidence_text":"", "evidence_spans_json":"[]", "evidence_source":"none",
                    "evidence_status":"not_mentioned_by_sentiment_model" if label == "NONE" else "pending_evidence",
                    "evidence_verified":"false", "evidence_needs_review":str(label != "NONE").lower(),
                    "evidence_display_mode":"none" if label == "NONE" else "full_review",
                    "evidence_quality_flags_json":"[]" if label == "NONE" else '["missing_evidence"]',
                    "evidence_model_version":"", "evidence_text_sha256":text_hash(text)}
        row.update(tagged_evidence(evidence, "model"))
    require(evidence["evidence_text_sha256"] == text_hash(text), "Bằng chứng ưu tiên khác văn bản nhận xét.")
    row.update(label=label, label_source=source, **evidence)
    return row


def validate_probability(row):
    supplied = [bool(str(row.get("p_" + x, "")).strip()) for x in LABELS]
    require(all(supplied) or not any(supplied), "Phân bố xác suất bị thiếu một phần; không tự điền giá trị.")
    for key in ("confidence", "margin"):
        if str(row.get(key, "")).strip():
            value = float(row[key])
            require(math.isfinite(value) and 0 <= value <= 1, "Điểm mô hình không hợp lệ: " + key)
    if not any(supplied):
        require(row.get("predicted_label") in LABELS, "Nhãn mô hình không hợp lệ.")
        return  # Đầu ra cũ không lưu phân bố; giữ trống thay vì tạo xác suất giả.
    probs = [float(row["p_"+x]) for x in LABELS]
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in probs) and abs(sum(probs)-1) < 1e-5, "Điểm mô hình không hợp lệ.")
    require(row["predicted_label"] == LABELS[max(range(5), key=lambda i: probs[i])], "Nhãn mô hình khác điểm đầu ra.")
    sorted_p = sorted(probs)
    for key, expected in (("confidence", sorted_p[-1]), ("margin", sorted_p[-1]-sorted_p[-2])):
        require(str(row.get(key, "")).strip() and abs(float(row[key])-expected) < 1e-6,
                "confidence/margin không khớp điểm đầu ra.")


def load_sources(root, cfg, paths):
    hashes = {k: sha(p) for k, p in paths.items()}
    dm, em, am = [jread(paths[k]) for k in ["data_manifest", "evidence_manifest", "a8_manifest"]]
    require(hashes["master"] == dm["output_sha256"]["reviews_master.csv"], "Corpus system không khớp data_manifest.")
    require(em["status"] == "complete" and em["signature"]["scope"] == "unlabeled" and em["signature"]["limit"] == 0,
            "Cần kết quả bằng chứng phần chưa gán tay đã chạy đầy đủ.")
    require(hashes["evidence"] == em["output_sha256"], "Tệp bằng chứng thiếu hoặc bị đổi so với prediction_manifest; dùng lại bản đầy đủ.")
    require(hashes["predictions"] == em["signature"]["predictions_sha256"], "Nguồn nhãn mô hình khác nguồn dùng khi lấy bằng chứng.")
    require(hashes["a8_alerts"] == am["output_sha256"]["canh_bao_a8.csv"], "Danh sách A8 đã bị đổi sau khi xuất.")
    require(hashes["a8_labels"] == am["labels_sha256"], "Bảng nhãn A8 khác danh sách đã xuất.")
    require(am["policy_version"] == "a8_system_rules_and_model_v01", "Chưa dùng danh sách A8 đã qua bước xử lý mới.")
    master = keyed(csv_rows(paths["master"]), "corpus")
    manual = keyed(csv_rows(paths["manual"]), "nhãn tay")
    a8_labels = keyed(csv_rows(paths["a8_labels"]), "nhãn A8")
    a8_alerts = keyed(csv_rows(paths["a8_alerts"]), "cảnh báo A8")
    require(len(master) == cfg["expected_counts"]["reviews"] == dm["corpus_rows"], "Sai số nhận xét corpus.")
    require(len(manual) == cfg["expected_counts"]["manual_reviews"] == dm["labeled_rows"], "Sai số nhận xét gán tay.")
    require(set(manual) <= set(master), "Nhãn tay có mã ngoài corpus.")
    for rid, row in master.items():
        require(row["review_text_nlp"].strip(), f"Corpus thiếu nội dung: {rid}")
        for field in ["restaurant_id", "restaurant_name", "review_date_iso", "city_std", "text_group_id"]:
            require(row.get(field, "").strip(), f"Corpus thiếu {field}: {rid}")
        parsed = datetime.fromisoformat(row["review_date_iso"].replace("Z", "+00:00"))
        require(parsed.tzinfo is not None, f"Ngày nhận xét không có múi giờ: {rid}")
        require(flag(row["has_manual_label"]) == (rid in manual), f"Cờ nhãn tay sai: {rid}")
        if rid in manual:
            require(manual[rid]["review_text_nlp"] == row["review_text_nlp"] and manual[rid]["text_group_id"] == row["text_group_id"], f"Nhãn tay không khớp corpus: {rid}")
            for a in ASPECTS:
                require(manual[rid][a] in LABELS, f"Nhãn tay sai ở {rid}/{a}")
    require(sum(flag(r["used_for_training"]) for r in master.values()) == dm["train_rows"], "Cờ huấn luyện không đúng số lượng.")
    require({rid for rid, r in master.items() if flag(r["excluded_vlsp_overlap"])} == set(dm["excluded_review_ids"]), "Hai trường hợp loại VLSP không khớp.")
    require(set(a8_labels) <= set(master) and set(a8_alerts) == {rid for rid, r in a8_labels.items() if r["A8_label"] == "1"}, "Danh sách A8 thiếu/thừa nhãn 1 đã chốt.")
    require(len(a8_alerts) == am["alerts"], "Sai số cảnh báo A8.")
    require(am["new_rule_and_model_alerts"] == 0, "Snapshot này yêu cầu danh sách A8 hiện tại chỉ gồm nhãn đã chốt.")
    for rid, label in a8_labels.items():
        require(label["A8_label"] in {"0", "1"} and label["review_text_nlp"] == master[rid]["review_text_nlp"], f"Nhãn A8 không khớp corpus: {rid}")
        for field in ["restaurant_id", "text_group_id"]:
            require(not label.get(field) or label[field] == master[rid][field], f"Liên kết A8 sai {field}: {rid}")
        if label["A8_label"] == "1":
            alert = a8_alerts[rid]
            require(alert["A8_label"] == "1" and alert["review_status"] == label["review_status"] and alert["review_text_nlp"] == master[rid]["review_text_nlp"], f"Trạng thái A8 không khớp: {rid}")
            require(alert["label_source"] == "NHAN_DA_CHOT", f"A8 mới chưa xác nhận lọt snapshot: {rid}")
            for span in json.loads(alert["A8_evidence_spans_json"]):
                require(master[rid]["review_text_nlp"][span["start"]:span["end"]] == span["text"], f"Đoạn A8 sai vị trí: {rid}")
    return hashes, master, manual, a8_labels, a8_alerts, em


def aspect_records(paths, master, manual, evidence_manifest, counts):
    """Ghép hai luồng CSV đã sắp xếp, không đưa 281 nghìn dòng vào bộ nhớ."""
    ev_iter = iter(csv_rows(paths["evidence"]))
    ev = next(ev_iter, None)
    previous, seen_mask = None, Counter()
    for model in csv_rows(paths["predictions"]):
        key = (model["review_id"], model["aspect"])
        require(previous is None or key > previous, "Dự đoán tổng phải duy nhất và sắp theo review_id, aspect.")
        previous = key
        rid, aspect = key
        require(rid in master and aspect in ASPECTS, f"Cặp ngoài corpus/A1–A7: {key}")
        require(model["label_source"] == "model", "Nguồn dự đoán tổng không phải model.")
        validate_probability(model)
        row = master[rid]
        for col in ["has_manual_label", "used_for_training", "excluded_vlsp_overlap"]:
            require(flag(model[col]) == flag(row[col]), f"Cờ dự đoán khác corpus: {key}/{col}")
        seen_mask[rid] += 1
        human = manual.get(rid)
        result = {"review_id": rid, "aspect": aspect, "label": human[aspect] if human else model["predicted_label"],
                  "label_source": "manual" if human else "model", "label_verified": str(bool(human) and not flag(human.get("uncertain", "0"))).lower(),
                  "manual_label": human[aspect] if human else "", "manual_uncertain": human.get("uncertain", "0") if human else "",
                  "model_label": model["predicted_label"], "model_version": model["model_version"],
                  **{f"p_{x}": model[f"p_{x}"] for x in LABELS},
                  "model_confidence": model["confidence"], "model_margin": model["margin"],
                  **{k: row[k] for k in ["has_manual_label", "used_for_training", "excluded_vlsp_overlap"]}}
        if human:
            result.update(manual_evidence(row["review_text_nlp"], human[aspect], human[aspect+"_evidence"], flag(human.get("uncertain", "0"))))
            counts["manual_pairs"] += 1
            counts["manual_model_disagreements"] += int(human[aspect] != model["predicted_label"])
        else:
            require(ev is not None and (ev["review_id"], ev["aspect"]) == key, f"Thiếu/trùng bằng chứng hoặc thứ tự sai: {key}")
            require(all(ev.get(k) == v for k, v in model.items()), f"Nhãn/điểm gốc bị đổi trong kết quả bằng chứng: {key}")
            result.update(model_evidence(row["review_text_nlp"], ev))
            counts["model_pairs"] += 1
            ev = next(ev_iter, None)
        counts["pairs"] += 1
        counts["label_source_"+result["label_source"]] += 1
        counts["evidence_status_"+result["evidence_status"]] += 1
        for f in json.loads(result["evidence_quality_flags_json"]):
            counts["evidence_flag_"+f] += 1
        retain_source_evidence(result, row["review_text_nlp"])
        yield result
    require(ev is None, "Kết quả bằng chứng còn dòng ngoài phần chưa gán tay.")
    require(set(seen_mask) == set(master) and set(seen_mask.values()) == {7}, "Mỗi nhận xét phải có đủ 7 khía cạnh duy nhất.")
    require(counts["model_pairs"] == evidence_manifest["output_rows"], "Số cặp bằng chứng không khớp báo cáo.")


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def temporal_coverage(master):
    parsed = pd.to_datetime(master.review_date_iso, errors="coerce", utc=True)
    if parsed.isna().any():
        bad = master.loc[parsed.isna(), "review_id"].head(10).tolist()
        raise ValueError(f"Ngày đăng thiếu/không hợp lệ: {bad}. Không tự thay bằng ngày thu thập.")
    work = master[["review_id", "restaurant_id", "restaurant_name", "text_group_id"]].copy()
    work["period"] = parsed.dt.tz_convert("Asia/Ho_Chi_Minh").dt.strftime("%Y-%m")
    work["date"] = parsed.dt.tz_convert("Asia/Ho_Chi_Minh").dt.strftime("%Y-%m-%d")
    result = work.groupby(["restaurant_id", "period"], sort=True).agg(
        restaurant_name=("restaurant_name", "first"), n_reviews=("review_id", "size"),
        n_text_groups=("text_group_id", "nunique"), first_date=("date", "min"),
        last_date=("date", "max")).reset_index()
    counts = {(r.restaurant_id, r.period): int(r.n_reviews) for r in result.itertuples()}
    for offset in (1, 2, 3):
        result[f"previous_{offset}_reviews"] = [counts.get(
            (r.restaurant_id, str(pd.Period(r.period, freq="M") - offset)), 0)
            for r in result.itertuples()]
    summary = {}
    for minimum in (5, 10, 20):
        eligible = result.n_reviews.ge(minimum)
        for offset in (1, 2, 3):
            eligible &= result[f"previous_{offset}_reviews"].ge(minimum)
        result[f"eligible_4_months_min_{minimum}"] = eligible
        summary[str(minimum)] = {"restaurant_months": int(eligible.sum()),
                                 "restaurants": int(result.loc[eligible, "restaurant_id"].nunique())}
    return result, summary


def prepare_base(cfg, overwrite=False, confirmed_override=None):
    out = location(cfg, "system_data_dir")
    reports = location(cfg, "system_report_dir")
    if (out / "data_manifest.json").exists() and not overwrite:
        raise FileExistsError("Đã có dữ liệu v01. Dùng --verify-only để kiểm tra; chỉ dùng --overwrite khi chủ động tạo lại.")
    paths = {"corpus": location(cfg, "corpus_file"), "labeled": location(cfg, "labeled_file"),
             "confirmed": resolve(cfg["_root"], confirmed_override) if confirmed_override
             else location(cfg, "confirmed_file")}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"Thiếu dữ liệu đầu vào: {p}")
    corpus, labeled, confirmed = [read_csv(paths[k]) for k in ("corpus", "labeled", "confirmed")]
    validate_reviews(corpus, "Corpus")
    validate_reviews(labeled, "Dữ liệu gán tay", True)
    validate_reviews(confirmed, "Dữ liệu thực nghiệm", True)
    require_columns(corpus, ["restaurant_id", "restaurant_name", "review_date_iso"], "Corpus")
    for df, key in [(corpus, "expected_corpus_rows"), (labeled, "expected_labeled_rows"),
                    (confirmed, "expected_train_rows")]:
        if len(df) != int(cfg[key]):
            raise ValueError(f"Số dòng không đúng {key}: thực tế {len(df)}, cấu hình {cfg[key]}.")
    excluded = set(cfg["excluded_review_ids"])
    if excluded != {"foody_18515513", "foody_3514027"}:
        raise ValueError("Danh sách loại trừ VLSP khác quyết định của v01.")
    if set(labeled.review_id) - excluded != set(confirmed.review_id):
        raise ValueError("3.348 nhận xét không đúng tập 3.350 trừ hai ID VLSP; không tự thêm/bớt dòng.")
    if not set(labeled.review_id).issubset(set(corpus.review_id)):
        raise ValueError("Có nhận xét gán tay không thuộc corpus.")
    for df, name in [(labeled, "gán tay"), (confirmed, "thực nghiệm")]:
        base = corpus.set_index("review_id")
        for col in ("review_text_nlp", "text_group_id"):
            if not df[col].eq(df.review_id.map(base[col])).all():
                bad = df.loc[~df[col].eq(df.review_id.map(base[col])), "review_id"].head(10).tolist()
                raise ValueError(f"{name}: {col} không khớp corpus: {bad}.")
    gold = labeled.set_index("review_id")
    for aspect in ASPECTS:
        if not confirmed[aspect].eq(confirmed.review_id.map(gold[aspect])).all():
            raise ValueError(f"Nhãn tay giữa hai bản không khớp ở {aspect}.")
    ref = reference_config(cfg)
    digest = fingerprint(confirmed)
    if digest != ref["dataset_sha256"]:
        raise ValueError("Văn bản/nhãn khác lần PhoBERT đã đánh giá. Không tiếp tục như cùng phiên bản.")
    train = confirmed.sort_values("review_id").reset_index(drop=True).copy()
    # Không lọc uncertain: giữ đúng tập đã thực nghiệm và ghi rõ số lượng.
    uncertain_ids = train.loc[train.get("uncertain", pd.Series("0", index=train.index)).eq("1"), "review_id"].tolist()
    master = corpus.sort_values("review_id").reset_index(drop=True).copy()
    master["has_manual_label"] = master.review_id.isin(set(labeled.review_id))
    master["used_for_training"] = master.review_id.isin(set(train.review_id))
    master["excluded_vlsp_overlap"] = master.review_id.isin(excluded)
    coverage, coverage_summary = temporal_coverage(master)
    warnings = []
    if uncertain_ids:
        warnings.append(f"Giữ nguyên {len(uncertain_ids)} nhận xét có uncertain=1 theo tập thực nghiệm; không tự đổi nhãn/cờ.")
    if not location(cfg, "preprocessing_file").exists():
        warnings.append("Chưa có tệp tách từ PhoBERT trong môi trường này; cần tệp gốc khi huấn luyện.")
    if not (location(cfg, "encoder_dir") / "config.json").exists():
        warnings.append("Checkpoint VLSP chưa có trong môi trường này; chuẩn bị dữ liệu vẫn thực hiện được.")
    audit = {"created_at": now(), "status": "passed", "corpus_rows": len(master),
             "labeled_rows": len(labeled), "train_rows": len(train),
             "unlabeled_rows": len(master) - len(labeled), "restaurants": int(master.restaurant_id.nunique()),
             "excluded_review_ids": sorted(excluded), "uncertain_train_rows": len(uncertain_ids),
             "uncertain_train_review_ids": uncertain_ids, "uncertain_policy": "preserved_as_in_experiment",
             "duplicate_text_group_rows": int(master.text_group_id.duplicated().sum()),
             "date_policy": "review_date_iso theo giờ Việt Nam; không dùng crawl_date thay ngày đăng",
             "temporal_coverage_three_prior_calendar_months": coverage_summary,
             "training_label_counts": {a: {label: int(train[a].eq(label).sum()) for label in LABELS} for a in ASPECTS},
             "warnings": warnings, "manual_labels_modified": False,
             "model_features": ["review_text_nlp"], "targets": ASPECTS}
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    atomic_csv(out / "train_reviews.csv", train)
    atomic_csv(out / "reviews_master.csv", master)
    atomic_csv(reports / "temporal_coverage.csv", coverage)
    atomic_json(reports / "data_audit.json", audit)
    inputs = {}
    for key, p in paths.items():
        try:
            displayed = str(p.relative_to(cfg["_root"])).replace("\\", "/")
        except ValueError:
            displayed = p.name
        inputs[key] = {"path": displayed, "sha256": sha256_file(p)}
    manifest = {"created_at": audit["created_at"], "system_version": cfg["system_version"],
                "corpus_rows": len(master), "labeled_rows": len(labeled), "train_rows": len(train),
                "unlabeled_rows": len(master) - len(labeled), "excluded_review_ids": sorted(excluded),
                "training_dataset_sha256": digest, "sources": inputs,
                "reference_run_config_sha256": sha256_file(location(cfg, "reference_run_config")),
                "output_sha256": {name: sha256_file(out / name)
                                  for name in ["train_reviews.csv", "reviews_master.csv"]},
                "label_policy": "Giữ 3348 nhận xét, không chuyển nhãn dự đoán thành nhãn tay",
                "training_input_features": ["review_text_nlp"], "targets": ASPECTS, "labels": LABELS}
    atomic_json(out / "data_manifest.json", manifest)
    print(json.dumps({k: audit[k] for k in ["status", "corpus_rows", "labeled_rows", "train_rows",
                                           "unlabeled_rows", "uncertain_train_rows", "warnings"]},
                     ensure_ascii=False, indent=2))
    print(f"Dữ liệu: {out}\nBáo cáo: {reports}", flush=True)
    return audit


# API cũ cho những kiểm thử hoặc mã chuẩn bị dữ liệu thực nghiệm đã có.
prepare = prepare_base


def input_configuration(cfg):
    require(cfg.get("system_schema_version") == SCHEMA,
            "Cần thay configs/system/foody_v01.json bằng cấu hình system chung đi kèm.")
    sources = dict(cfg["system_input_sources"])
    # Hai nguồn nền luôn nằm tại chính system hiện có, không đọc foody_inputs_v02.
    sources["master"] = [cfg["system_data_dir"] + "/reviews_master.csv"]
    sources["data_manifest"] = [cfg["system_data_dir"] + "/data_manifest.json"]
    return {"sources": sources, "expected_counts": {"reviews": cfg["expected_corpus_rows"],
            "manual_reviews": cfg["expected_labeled_rows"]}}


def validate_policy(cfg):
    names = [cfg["system_aspects_file"], cfg["system_a8_file"], cfg["system_inputs_manifest"]]
    protected = {"reviews_master.csv", "train_reviews.csv", "data_manifest.json", "phobert_text_cache.sqlite3"}
    require(len(set(names)) == 3 and all(Path(n).name == n and n not in protected for n in names),
            "Tên đầu ra không được trùng tệp nền hoặc chứa đường dẫn thư mục.")
    policy = cfg["alert_policy"]
    content, trend = policy["content"], policy["trend"]
    require(set(content["reflection_labels"]) == {"NEG", "MIX"}, "Nhãn phản ánh không đúng bộ đã chốt.")
    require(trend["baseline_previous_calendar_periods"] == 3 and trend["manual_labels_replace_predictions"] is True,
            "Cấu hình nền cần giữ mốc cơ sở và bật ưu tiên nhãn tay đã xác nhận.")
    require(trend["model_confidence_filter"] is None, "Không tự thêm lọc điểm vào mẫu số xu hướng.")
    require(trend["baseline_implementation"] == "arithmetic_mean_of_three_calendar_period_rates", "Công thức đường cơ sở đã đổi.")
    require(not policy["a8"]["use_legacy_incident_rules"], "Không dùng lại nhánh A8 cũ.")
    require(all(r["aspect"] in ASPECTS for r in content["aspect_rules"]), "Luật mức độ có khía cạnh ngoài A1–A7.")


def system_paths(cfg):
    """Đường dẫn bản vận hành đang dùng; tập đã huấn luyện luôn ở nền gốc."""
    from src.pipeline.update_system_data import current_paths
    return current_paths(cfg)


def verify_system(cfg, check_sources=True):
    """Kiểm tra cả nền huấn luyện bất biến và đầu vào vận hành đang dùng."""
    paths = system_paths(cfg)
    if paths["directory"] != location(cfg, "system_data_dir"):
        from src.pipeline.update_system_data import verify_snapshot
        return verify_snapshot(cfg, paths, check_sources)
    train, master, base = prepared_data(cfg)
    out = location(cfg, "system_data_dir")
    path = out / cfg["system_inputs_manifest"]
    manifest = jread(path)
    require(manifest.get("status") == "complete" and manifest.get("schema_version") == SCHEMA,
            "System chưa cập nhật hoàn tất. Chạy --stage update.")
    for name, digest in manifest["output_sha256"].items():
        target = out / name
        require(target.parent == out and target.is_file() and sha(target) == digest,
                f"Đầu ra system thiếu/đã bị sửa: {target}")
    require(manifest["base_identity"]["master_sha256"] == sha(out/"reviews_master.csv")
            and manifest["base_identity"]["train_sha256"] == sha(out/"train_reviews.csv")
            and manifest["base_identity"]["training_dataset_sha256"] == base["training_dataset_sha256"],
            "Nền corpus/huấn luyện khác bản đang dùng cho system.")
    if check_sources:
        paths = source_paths(cfg["_root"], input_configuration(cfg))
        require(all(sha(p) == manifest["signature"]["sources"][k] for k, p in paths.items()),
                "Nguồn nhãn/bằng chứng/A8 đã cập nhật. Chạy --stage update để ghi lại system tại chỗ.")
        require(sha(cfg["_config"]) == manifest["signature"]["config_sha256"],
                "Cấu hình system đã đổi. Chạy --stage update.")
    return manifest


def report_system(cfg, manifest):
    reports = location(cfg, "system_report_dir")
    reports.mkdir(parents=True, exist_ok=True)
    target = reports / "data_audit.json"
    audit = jread(target) if target.is_file() else {}
    audit.update(status="passed", system_schema_version=SCHEMA,
                 latest_system_update=manifest["created_at_utc"],
                 system_update=manifest["summary"],
                 base_training_snapshot_unchanged=True,
                 source_paths=manifest["source_paths"],
                 output_sha256=manifest["output_sha256"])
    atomic_json(target, audit)
    summary = manifest["summary"]
    text = "\n".join([
        "# System chung", "",
        "Cấu hình duy nhất: configs/system/foody_v01.json.",
        "Đầu vào vận hành nằm cùng corpus ở data/system/foody_v01; báo cáo tại reports/system/foody_v01.",
        "Không cần foody_inputs_v02 hoặc các thư mục alerts cũ.", "",
        "## Kết quả cập nhật", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
        "## Cách dùng", "",
        "reviews_master.csv giữ nguyên dữ liệu nền và thông tin quán. train_reviews.csv giữ nguyên tập đã dùng huấn luyện.",
        "aspect_inputs.csv dùng label ưu tiên nhãn tay; model_label và điểm máy giữ riêng. Bằng chứng có nguồn và trạng thái riêng.",
        "a8_inputs.csv chứa cảnh báo A8 đã xử lý. Nhãn A8 chưa duyệt không tự trở thành nhãn 0.",
        "Các cờ bằng chứng chỉ phát hiện lỗi hình thức, không xác nhận mô hình trích đúng ngữ nghĩa.", "",
        "## Cập nhật lần sau", "",
        "Sau khi thay kết quả bằng chứng và prediction_manifest.json tương ứng tại nguồn, chạy lại python -m src.pipeline.prepare_system_data.",
        "Chương trình tự ghi đè đúng đầu ra vận hành trong system hiện có. Không tạo thêm phiên bản thư mục, không huấn luyện lại.",
        "Dữ liệu/mô hình nguồn vẫn cần được giữ ở chỗ cũ. Chưa dựng lại các tầng cảnh báo trong bước này.", ""])
    temp = reports / "kiem_tra_system.md.tmp"
    temp.write_text(text, encoding="utf-8")
    temp.replace(reports / "kiem_tra_system.md")


def replace_generated(out, staging, names):
    """Thay các tệp sinh tự động; nếu lỗi giữa chừng thì hoàn trả bản trước."""
    rollback = staging / "rollback"
    rollback.mkdir()
    existed = {}
    for name in names:
        target = out / name
        existed[name] = target.is_file()
        if target.is_file():
            shutil.copy2(target, rollback / name)
    committed = []
    try:
        for name in names:
            os.replace(staging/name, out/name)
            committed.append(name)
    except BaseException:
        for name in reversed(committed):
            if existed[name]:
                os.replace(rollback/name, out/name)
            else:
                (out/name).unlink(missing_ok=True)
        raise


def update_system(cfg, audit_only=False):
    """Cập nhật tại foody_v01. Mọi kiểm tra hoàn thành trước khi thay đầu ra cũ."""
    require(system_paths(cfg)["directory"] == location(cfg, "system_data_dir"),
            "Đã dùng cập nhật theo đợt. Nhập đợt mới bằng src.pipeline.update_system_data; không ghi lại nguồn 3.350 cũ.")
    validate_policy(cfg)
    root, out = cfg["_root"], location(cfg, "system_data_dir")
    train, base_master, base = prepared_data(cfg)
    # Tập huấn luyện được bảo vệ bởi mã kiểm tra/fingerprint riêng, không dùng nhãn máy để sửa.
    training_labels = {str(r.review_id): {a: str(getattr(r, a)) for a in ASPECTS} for r in train.itertuples(index=False)}
    del train, base_master
    input_cfg = input_configuration(cfg)
    paths = source_paths(root, input_cfg)
    hashes, master, manual, a8_labels, a8_alerts, em = load_sources(root, input_cfg, paths)
    for p in paths.values():
        require(p not in {out/cfg["system_aspects_file"], out/cfg["system_a8_file"], out/cfg["system_inputs_manifest"]},
                "Nguồn đầu vào không được trỏ vào chính đầu ra system.")
    human_changes = sum(manual[rid][a] != labels[a] for rid, labels in training_labels.items() for a in ASPECTS)
    signature = {"version": VERSION, "schema_version": SCHEMA,
                 "config_sha256": cfg["_config_sha256"], "program_sha256": sha(__file__),
                 "common_program_sha256": sha(Path(__file__).parents[1]/"modeling/phobert_system_common.py"),
                 "sources": hashes}
    base_identity = {"master_sha256": hashes["master"], "train_sha256": sha(out/"train_reviews.csv"),
                     "training_dataset_sha256": base["training_dataset_sha256"]}
    if audit_only:
        print(json.dumps({"status": "sources_validated", "reviews": len(master), "manual_reviews": len(manual),
                          "a8_alerts": len(a8_alerts), "system_data_dir": str(out),
                          "requires_legacy_alerts": False, "source_paths": {k: str(p) for k,p in paths.items()}},
                         ensure_ascii=False, indent=2), flush=True)
        return None
    previous = out / cfg["system_inputs_manifest"]
    if previous.is_file():
        saved = jread(previous)
        if saved.get("signature") == signature:
            verify_system(cfg)
            report_system(cfg, saved)
            print("System đã cập nhật, nguồn không đổi:", out, flush=True)
            print(json.dumps(saved["summary"], ensure_ascii=False, indent=2))
            return saved
    staging = Path(tempfile.mkdtemp(prefix="system_update_", dir=out.parent))
    counts = Counter()
    try:
        print("Đang cập nhật A1–A7 và bằng chứng vào system hiện có...", flush=True)
        write_csv(staging/cfg["system_aspects_file"], ASPECT_FIELDS, aspect_records(paths, master, manual, em, counts))
        require(counts["pairs"] == len(master)*7, "Thiếu cặp khía cạnh sau ghép.")
        def a8_records():
            for rid in sorted(a8_alerts):
                row = dict(a8_alerts[rid])
                row.update({k: master[rid][k] for k in ["restaurant_id", "restaurant_name", "review_date_iso", "city_std", "text_group_id", "review_text_nlp"]})
                yield row
        write_csv(staging/cfg["system_a8_file"], A8_FIELDS, a8_records())
        require(all(sha(p) == hashes[k] for k,p in paths.items()) and sha(out/"train_reviews.csv") == base_identity["train_sha256"],
                "Nguồn đã đổi trong lúc cập nhật. Giữ nguyên system cũ; chạy lại khi nguồn đã ổn định.")
        require(sha(cfg["_config"]) == signature["config_sha256"] and sha(__file__) == signature["program_sha256"],
                "Cấu hình/mã đã đổi trong lúc cập nhật. Giữ nguyên system cũ và chạy lại.")
        summary = {"reviews": len(master), "restaurants": len({r["restaurant_id"] for r in master.values()}),
                   "manual_reviews": len(manual), "model_only_reviews": len(master)-len(manual),
                   "aspect_pairs": counts["pairs"], "manual_aspect_pairs": counts["manual_pairs"],
                   "model_aspect_pairs": counts["model_pairs"], "manual_model_label_disagreements": counts["manual_model_disagreements"],
                   "operational_manual_label_changes_vs_training": human_changes,
                   "a8_alerts": len(a8_alerts), "a8_by_status": dict(Counter(r["review_status"] for r in a8_alerts.values())),
                   "a8_explicit_negative_reviews": sum(r["A8_label"] == "0" for r in a8_labels.values()),
                   "a8_unlabeled_reviews": len(master)-len(a8_labels),
                   "evidence_status_counts": {k.removeprefix("evidence_status_"): v for k,v in sorted(counts.items()) if k.startswith("evidence_status_")},
                   "evidence_quality_flag_counts": {k.removeprefix("evidence_flag_"): v for k,v in sorted(counts.items()) if k.startswith("evidence_flag_")},
                   "base_files_modified": False, "model_retrained": False, "alerts_rebuilt": False,
                   "new_independent_model_evaluation": False, "requires_legacy_alerts": False}
        manifest = {"status": "complete", "schema_version": SCHEMA, "version": VERSION,
                    "created_at_utc": now(), "signature": signature, "base_identity": base_identity,
                    "summary": summary, "source_paths": {k: str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for k,p in paths.items()},
                    "a8_known_negative_review_ids": sorted(rid for rid,r in a8_labels.items() if r["A8_label"] == "0"),
                    "operational_label_column": "label", "trend_label_column": "model_label",
                    "evidence_offset_reference": "reviews_master.review_text_nlp; Unicode code points; [start,end)",
                    "output_sha256": {name: sha(staging/name) for name in [cfg["system_aspects_file"], cfg["system_a8_file"]]}}
        atomic_json(staging/cfg["system_inputs_manifest"], manifest)
        replace_generated(out, staging, [cfg["system_aspects_file"], cfg["system_a8_file"], cfg["system_inputs_manifest"]])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    report_system(cfg, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Đã cập nhật tại:", out, flush=True)
    return manifest


def load_system_inputs(project_root=ROOT, view="operational", config=None, snapshot_dir=None, allow_config_change=False):
    """Đầu đọc chung cho giao diện và bộ tạo cảnh báo, không cần thư mục phiên bản phụ."""
    require(view in {"operational", "trend"}, "view phải là operational hoặc trend")
    cfg = load_settings(config, project_root)
    cfg["_allow_config_change"] = allow_config_change
    if snapshot_dir is not None:
        cfg["_system_snapshot"] = Path(snapshot_dir)
    # Chốt một đường dẫn cho cả lần đọc; cập nhật đồng thời không trộn hai bản.
    paths = system_paths(cfg)
    cfg["_system_snapshot"] = paths["directory"]
    manifest = verify_system(cfg, check_sources=False)
    require(view != "trend" or manifest["summary"].get("pending_model_reviews", 0) == 0,
            "Chưa đủ dự đoán máy cho toàn bộ corpus. Tầng 1 vẫn dùng nhãn tay; chưa tính tầng 2 bằng tập thiếu dự đoán.")
    master = keyed(csv_rows(paths["master"]), "master")
    a8 = list(csv_rows(paths["a8"]))
    positives = {r["review_id"]: r for r in a8}
    negatives = set(manifest["a8_known_negative_review_ids"])
    for rid, r in master.items():
        r.update(A8_label="1" if rid in positives else "0" if rid in negatives else "",
                 A8_review_status=positives[rid]["review_status"] if rid in positives else "DA_DUYET" if rid in negatives else "CHUA_GAN_NHAN")
    def aspects():
        for r in csv_rows(paths["aspects"]):
            retain_source_evidence(r, master[r["review_id"]]["review_text_nlp"])
            confirmed = r["label_source"] == "manual" and flag(r["label_verified"])
            r["predicted_label"] = (r.get("manual_label") or r["label"]) if confirmed else r["model_label"]
            r["selected_label_source"] = "manual" if confirmed else "model"
            r["confidence"] = "" if confirmed else r["model_confidence"]
            r["margin"] = "" if confirmed else r["model_margin"]
            yield r
    return {"master": master, "aspects": aspects(), "a8": a8, "manifest": manifest, "paths": paths}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--project-root", default=None)
    p.add_argument("--stage", choices=["update", "audit", "base", "verify"], default="update")
    p.add_argument("--verify-only", action="store_true", help="Tương đương --stage verify")
    p.add_argument("--overwrite", action="store_true", help="Chỉ dùng với --stage base khi chủ động tạo lại nền corpus/huấn luyện")
    p.add_argument("--confirmed-file", default=None)
    args = p.parse_args(argv)
    cfg = load_settings(args.config, args.project_root)
    if args.stage == "base":
        prepare_base(cfg, args.overwrite, args.confirmed_file)
        print("Đã chuẩn bị nền; chạy lại không có --stage base để cập nhật đầu vào vận hành.")
        return 0
    if args.verify_only or args.stage == "verify":
        result = verify_system(cfg)
        print("System hợp lệ:", json.dumps(result["summary"], ensure_ascii=False))
        return 0
    if not (location(cfg,"system_data_dir")/"data_manifest.json").is_file():
        prepare_base(cfg, False, args.confirmed_file)
    update_system(cfg, audit_only=args.stage == "audit")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, json.JSONDecodeError) as exc:
        print("DỪNG:", exc, file=sys.stderr, flush=True)
        raise SystemExit(2)

