"""Nhập dữ liệu theo đợt; tạo system và các tầng cảnh báo trước khi kích hoạt.

Không huấn luyện, không sửa corpus/tập học đã đóng băng.
Tự cập nhật tầng 2 khi configs/alerts/tang2/trend_alerts.json đã bật.
Mặc định chỉ kiểm tra; --apply mới kích hoạt kết quả. Xem hướng dẫn đi kèm.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from zoneinfo import ZoneInfo

from src.modeling.phobert_system_common import ASPECTS, LABELS, load_settings, location, now, prepared_data
from src.pipeline.prepare_system_data import (
    ASPECT_FIELDS, A8_FIELDS, atomic_json, compact, csv_rows, flag, jread, keyed,
    manual_evidence, model_evidence, require, sha, text_hash, validate_probability,
    EVIDENCE_FIELDS, SOURCE_EVIDENCE_COLUMNS, SOURCE_EVIDENCE_SCHEMA,
    tagged_evidence, source_evidence, retain_source_evidence, select_operational_source,
    LAYOUT_VERSION, safe_name, system_output_root, alert_output_root, alert_output_dir,
)


VERSION = "foody_system_updates_v01"
POINTER = "current_update.json"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
INCIDENTS = {"SUC_KHOE", "PHAN_BIET_DOI_XU", "XUC_PHAM_NGHIEM_TRONG",
             "AN_TOAN_TAI_CO_SO", "TRAI_NGHIEM_CUC_TIEU_CUC", "BAO_LUC_DE_DOA"}
SOURCE_KEYS = ("reviews_file", "manual_file", "predictions_file", "evidence_file", "a8_file",
               "model_profile_file", "evidence_profile_file", "source_manifest_file")


def safe_id(value):
    require(isinstance(value, str) and SAFE_ID.fullmatch(value) and value not in {".", ".."},
            "batch_id chỉ dùng chữ không dấu, số, dấu gạch hoặc dấu chấm; tối đa 80 ký tự.")
    return value


def current_paths(cfg):
    base = location(cfg, "system_data_dir")
    override = cfg.get("_system_snapshot")
    if override:
        out = Path(override).resolve()
    elif (system_output_root(cfg) / POINTER).is_file() or (base / POINTER).is_file():
        report_root = system_output_root(cfg)
        pointer_path = report_root / POINTER if (report_root / POINTER).is_file() else base / POINTER
        pointer = jread(pointer_path)
        parent = report_root if pointer_path.parent == report_root else base / "updates"
        out = parent / safe_id(pointer["batch_id"])
        require(out.resolve().is_relative_to(parent.resolve()), "Đường dẫn bản cập nhật không hợp lệ.")
        require(out.is_dir(), "Bản system đang dùng bị thiếu; không tự quay về bản cũ.")
        require(sha(out / cfg["system_inputs_manifest"]) == pointer["manifest_sha256"],
                "Thông tin kích hoạt không khớp system; cần kiểm tra lại đợt cập nhật.")
    else:
        out = base
    return {"directory": out, "master": out / "reviews_master.csv",
            "aspects": out / cfg["system_aspects_file"], "a8": out / cfg["system_a8_file"],
            "manifest": out / cfg["system_inputs_manifest"]}


def verify_snapshot(cfg, paths, check_sources=True):
    manifest = jread(paths["manifest"])
    require(manifest.get("version") == VERSION and manifest.get("status") == "complete"
            and manifest.get("schema_version") == 2, "Bản cập nhật system chưa hoàn tất.")
    _, _, base_manifest = prepared_data(cfg)
    base = location(cfg, "system_data_dir")
    identity = manifest["base_identity"]
    require(identity["master_sha256"] == sha(base / "reviews_master.csv") and
            identity["train_sha256"] == sha(base / "train_reviews.csv") and
            identity["training_dataset_sha256"] == base_manifest["training_dataset_sha256"],
            "Tập nền hoặc tập đã huấn luyện đã đổi; không trộn vào đợt cập nhật.")
    if not cfg.get("_allow_config_change"):
        require(manifest["signature"]["config_sha256"] == sha(cfg["_config"]),
                "Cấu hình đã đổi so với system; cần tạo đợt cập nhật phù hợp.")
    for name, digest in manifest["output_sha256"].items():
        target = (paths["directory"] / name).resolve()
        require(target.is_relative_to(paths["directory"].resolve()) and target.is_file()
                and sha(target) == digest, "Tệp cập nhật thiếu hoặc bị sửa: " + name)
    if manifest.get("source_evidence_schema_version"):
        require(manifest["source_evidence_schema_version"] == SOURCE_EVIDENCE_SCHEMA, "Chưa hỗ trợ cấu trúc bằng chứng nguồn này.")
        with paths["aspects"].open(encoding="utf-8-sig", newline="") as stream:
            require(set(ASPECT_FIELDS) <= set(csv.reader(stream).__next__()), "System bị thiếu các cột bằng chứng máy/tay riêng.")
    # Nguồn mỗi đợt đã được chép vào inputs và kiểm tra hash. Nguồn ở ngoài có
    # thể được di chuyển/chỉnh sửa sau đó mà không làm thay đổi bản đã kích hoạt.
    return manifest


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def rows_file(path):
    if path is None:
        return []
    return list(csv_rows(path))


def canonical_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def review_period(row):
    date = datetime.fromisoformat(row["review_date_iso"].replace("Z", "+00:00"))
    require(date.tzinfo is not None, "Ngày đăng phải có múi giờ: " + row["review_id"])
    return date.astimezone(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m")


def next_periods(period):
    year, month = map(int, period.split("-"))
    return [f"{(year * 12 + month - 1 + i) // 12:04d}-{(year * 12 + month - 1 + i) % 12 + 1:02d}"
            for i in range(4)]


def read_batch(path):
    path = Path(path).resolve()
    data = jread(path)
    safe_id(data.get("batch_id"))
    require(not (set(data) - {"batch_id", "annotation_version", "manual_confirmed", "replace_manual",
                              "replace_predictions", "note", *SOURCE_KEYS}), "Có khóa không được hỗ trợ trong tệp đợt cập nhật.")
    sources = {}
    for key in SOURCE_KEYS:
        if data.get(key):
            source = Path(data[key])
            source = source.resolve() if source.is_absolute() else (path.parent / source).resolve()
            require(source.is_file(), "Thiếu đầu vào: " + str(source))
            suffix = ".json" if key.endswith(("profile_file", "manifest_file")) else ".csv"
            require(source.suffix.casefold() == suffix, "Đầu vào sai định dạng " + suffix + ": " + str(source))
            sources[key] = source
    require(sources, "Đợt cập nhật chưa chỉ định tệp dữ liệu nào.")
    for key in ("manual_confirmed", "replace_manual", "replace_predictions"):
        require(key not in data or type(data[key]) is bool, key + " phải là true/false trong JSON.")
    hashes = {key: sha(value) for key, value in sources.items()}
    identity = hashlib.sha256(compact({"definition": data, "source_hashes": hashes}).encode()).hexdigest()
    return {"definition": data, "path": path, "sources": sources, "hashes": hashes,
            "identity": identity, "batch_id": data["batch_id"]}


def normalize_review(row, fields, groups, training_ids, excluded_ids):
    for key in ("review_id", "restaurant_id", "restaurant_name", "review_text_nlp", "review_date_iso", "city_std"):
        require(row.get(key, "").strip(), "Nhận xét mới thiếu " + key)
    rid, text = row["review_id"], row["review_text_nlp"]
    require(text == unicodedata.normalize("NFC", text), "Chuẩn hóa Unicode NFC trước khi gán nhãn: " + rid)
    require("\x00" not in text, "Nội dung chứa ký tự NUL: " + rid)
    period = review_period(row)
    group_key = canonical_text(text)
    proposed_group = row.get("text_group_id", "").strip()
    known_group = groups.get(group_key)
    require(not (known_group and proposed_group and known_group != proposed_group),
            "Nhận xét trùng nội dung phải dùng nhóm văn bản đã có: " + rid)
    group = known_group or proposed_group or hashlib.sha1(group_key.encode()).hexdigest()
    groups[group_key] = group
    result = {key: row.get(key, "") for key in fields}
    result.update(text_group_id=group, review_text=row.get("review_text") or text,
                  review_length_chars=str(len(text)), review_year=period[:4], review_month=period,
                  has_manual_label="false", used_for_training=str(rid in training_ids).lower(),
                  excluded_vlsp_overlap=str(rid in excluded_ids).lower())
    return result


def manual_changes(batch, master):
    changes = {}
    for row in rows_file(batch["sources"].get("manual_file")):
        rid = row["review_id"]
        require(rid in master, "Nhãn tay có mã ngoài corpus; cần nhập reviews_file trước: " + rid)
        require(row.get("review_text_nlp") == master[rid]["review_text_nlp"], "Nhãn tay dùng văn bản khác: " + rid)
        require(not row.get("text_group_id") or row["text_group_id"] == master[rid]["text_group_id"], "Nhóm văn bản của nhãn tay không khớp.")
        require(row.get("label_source", "manual") != "model", "Không nhập dự đoán làm nhãn tay.")
        confirmed = batch["definition"].get("manual_confirmed", False) or row.get("label_status") in {
            "Hoàn thành", "DA_DUYET", "DA_XAC_NHAN", "CONFIRMED"}
        require(confirmed, "Nhãn tay phải có trạng thái xác nhận hoặc manual_confirmed=true: " + rid)
        uncertain = flag(row.get("uncertain", "0"))
        if "aspect" in row:
            entries = [(row["aspect"], row.get("label", ""), row.get("evidence_text", ""))]
        else:
            require(all(a in row and a + "_evidence" in row for a in ASPECTS), "Bảng nhãn tay dạng rộng phải có đủ A1–A7 và các cột bằng chứng.")
            entries = [(a, row[a], row[a + "_evidence"]) for a in ASPECTS]
        for aspect, label, evidence in entries:
            require(aspect in ASPECTS and label in LABELS, "Nhãn tay ngoài bộ khía cạnh/nhãn.")
            key = (rid, aspect)
            require(key not in changes, "Trùng cặp nhãn tay trong cùng đợt: " + str(key))
            # Dữ liệu đã xác nhận có thể chứa nhiều đoạn nguyên văn; giữ đủ nguồn.
            ev = manual_evidence(master[rid]["review_text_nlp"], label, evidence, uncertain)
            explicit = row.get("evidence_spans_json", "") if "aspect" in row else row.get(aspect + "_evidence_spans_json", "")
            if explicit:
                spans = json.loads(explicit)
                check_spans(master[rid]["review_text_nlp"], spans)
                require([s["text"].strip() for s in spans] == [s.strip() for s in evidence.split("||") if s.strip()], "Vị trí bằng chứng tay không khớp nội dung đã xác nhận.")
                ev.update(evidence_spans_json=compact(spans), evidence_verified=str(not uncertain).lower(),
                          evidence_needs_review=str(uncertain).lower(), evidence_status="manual_confirmed" if not uncertain else "manual_evidence_needs_review",
                          evidence_quality_flags_json=compact(["manual_annotation_marked_uncertain"] if uncertain else []),
                          evidence_display_mode="quoted_spans" if label != "NONE" else "none")
            changes[key] = {"label": label, "label_source": "manual", "label_verified": str(not uncertain).lower(),
                            "manual_label": label, "manual_uncertain": str(int(uncertain)), **ev,
                            **tagged_evidence(ev, "manual")}
    return changes


def check_spans(text, spans):
    require(isinstance(spans, list), "Danh sách vị trí bằng chứng phải là mảng JSON.")
    for span in spans:
        start, end = span["start"], span["end"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                and text[start:end] == span["text"], "Bằng chứng không khớp vị trí trong văn bản gốc.")


def prediction_changes(rows, master, model_version):
    changes, coverage = {}, Counter()
    for row in rows:
        rid, aspect = row["review_id"], row.get("aspect")
        require(rid in master and aspect in ASPECTS, "Dự đoán nằm ngoài corpus/A1–A7.")
        key = (rid, aspect)
        require(key not in changes, "Trùng cặp dự đoán: " + str(key))
        require(row.get("model_version") == model_version and row.get("label_source") == "model",
                "Không trộn phiên bản mô hình hoặc nhập nhãn tay vào nguồn dự đoán.")
        require(row.get("review_text_sha256") == text_hash(master[rid]["review_text_nlp"]) or
                row.get("review_text_nlp") == master[rid]["review_text_nlp"], "Dự đoán thiếu bằng chứng liên kết đúng văn bản: " + rid)
        validate_probability(row)
        changes[key] = {"model_label": row["predicted_label"], "model_version": model_version,
                        **{f"p_{x}": row.get(f"p_{x}", "") for x in LABELS},
                        "model_confidence": row.get("confidence", ""), "model_margin": row.get("margin", ""),
                        "review_text_sha256": text_hash(master[rid]["review_text_nlp"])}
        coverage[rid] += 1
    require(all(n == 7 for n in coverage.values()), "Mỗi nhận xét trong tệp dự đoán cần đủ bảy khía cạnh.")
    return changes


def empty_aspect(master, aspect):
    row = {key: "" for key in ASPECT_FIELDS}
    row.update(review_id=master["review_id"], aspect=aspect, label_source="model", label_verified="false",
               evidence_text="", evidence_spans_json="[]", evidence_source="none", evidence_status="pending_evidence",
               evidence_verified="false", evidence_needs_review="true", evidence_display_mode="full_review",
               evidence_quality_flags_json='["missing_evidence"]', evidence_text_sha256=text_hash(master["review_text_nlp"]))
    return row


def clear_model_evidence(row, text):
    none = row["model_label"] == "NONE"
    cleared = dict(evidence_text="", evidence_spans_json="[]", evidence_source="none",
               evidence_status="not_mentioned_by_sentiment_model" if none else "pending_evidence",
               evidence_verified="false", evidence_needs_review=str(not none).lower(),
               evidence_display_mode="none" if none else "full_review",
               evidence_quality_flags_json="[]" if none else '["missing_evidence"]',
               evidence_model_version="", evidence_text_sha256=text_hash(text))
    row.update(tagged_evidence(cleared, "model"))
    if row["label_source"] != "manual":row.update(cleared)


def prepare_batch(cfg, batch_path):
    from src.pipeline.prepare_system_data import load_system_inputs
    batch = read_batch(batch_path)
    inputs = load_system_inputs(cfg["_root"], config=cfg["_config"], allow_config_change=True)
    paths, manifest = inputs["paths"], inputs["manifest"]
    history = manifest.get("update_history", [])
    previous = next((b for b in history if b["batch_id"] == batch["batch_id"]), None)
    if previous:
        require(previous["identity"] == batch["identity"], "batch_id đã được dùng với nội dung khác; tạo mã đợt mới.")
        require(manifest["signature"]["config_sha256"] == sha(cfg["_config"]),
                "Cấu hình đã đổi; dùng mã đợt mới để cập nhật System trước khi chạy cảnh báo.")
        return {"already_applied": True, "batch": batch, "manifest": manifest}
    train, _, base_manifest = prepared_data(cfg)
    training_ids = set(train.review_id)
    excluded_ids = set(base_manifest["excluded_review_ids"])
    master = inputs["master"]
    for row in master.values():
        row.pop("A8_label", None); row.pop("A8_review_status", None)
    fields = list(next(iter(master.values())))
    reviews = rows_file(batch["sources"].get("reviews_file"))
    require(len({r["review_id"] for r in reviews}) == len(reviews), "Trùng mã trong tệp nhận xét mới.")
    groups = {canonical_text(r["review_text_nlp"]): r["text_group_id"] for r in master.values()}
    new_ids = set()
    for source in reviews:
        for key in source:
            if key not in fields and key not in {"A8_label", "A8_review_status"}:
                fields.append(key)
        normalized = normalize_review(source, fields, groups, training_ids, excluded_ids)
        rid = source["review_id"]
        if rid in master:
            for key in ("restaurant_id", "restaurant_name", "review_text_nlp", "review_date_iso", "city_std", "text_group_id"):
                require(normalized[key] == master[rid][key], "Mã đã có nhưng dữ liệu khác ở " + key + ": " + rid)
        else:
            master[rid] = normalized
            new_ids.add(rid)
    manual = manual_changes(batch, master)
    for rid, _ in manual:
        master[rid]["has_manual_label"] = "true"
    model_version = manifest.get("current_model_version", cfg["model_version"])
    profile = manifest.get("runtime_model")
    changing_model = False
    if batch["sources"].get("model_profile_file"):
        from src.modeling.runtime_model import model_bundle
        supplied_profile = jread(batch["sources"]["model_profile_file"])
        _, _, _, training_ids, profile = model_bundle(cfg, supplied_profile)
        require(training_ids <= set(master), "Mô hình có ID đã học ngoài tập review hệ thống; nhập review trước hoặc trong cùng đợt.")
        # excluded_ids chỉ đánh dấu mẫu không dùng chấm CV; mô hình cuối được học chúng.
        changing_model = profile["model_version"] != model_version
        require(changing_model or profile == manifest.get("runtime_model"),
                "Dùng mã phiên mô hình mới khi đăng ký mô hình; không ghi đè mô hình cùng mã.")
        require(not changing_model or batch["definition"].get("replace_predictions", False),
                "Thay mô hình cần replace_predictions=true và dự đoán lại toàn bộ corpus.")
        model_version = profile["model_version"]
    elif profile:
        training_ids = set(profile["training_review_ids"])
    predictions = prediction_changes(rows_file(batch["sources"].get("predictions_file")), master, model_version)
    evidence_rows = rows_file(batch["sources"].get("evidence_file"))
    evidence = {}
    for row in evidence_rows:
        key = (row["review_id"], row.get("aspect"))
        require(key[0] in master and key[1] in ASPECTS and key not in evidence, "Bằng chứng trùng cặp hoặc ngoài corpus.")
        require(row.get("model_version") == model_version, "Bằng chứng thuộc mô hình gán nhãn khác.")
        evidence[key] = row
    targets = set(manual) | set(predictions) | set(evidence)
    old = {}
    missing_models, model_counts = set(new_ids), Counter()
    for row in inputs["aspects"]:
        rid, aspect = row["review_id"], row["aspect"]
        if row["model_label"] and not changing_model:
            model_counts[rid] += 1
        if (rid, aspect) in targets:
            old[(rid, aspect)] = {k: row.get(k, "") for k in ASPECT_FIELDS}
    missing_models.update(rid for rid in master if model_counts[rid] != 7)
    supplied_ids = {rid for rid, _ in predictions}
    missing_models -= supplied_ids
    for key, change in manual.items():
        existing = old.get(key)
        if existing and existing["label_source"] == "manual" and any(existing.get(k) != v for k, v in change.items()):
            require(batch["definition"].get("replace_manual", False), "Đang sửa nhãn/bằng chứng tay đã có; cần replace_manual=true: " + str(key))
    for key, change in predictions.items():
        existing = old.get(key)
        if existing and existing["model_label"] and any(existing.get(k) != str(v) for k, v in change.items()):
            require(batch["definition"].get("replace_predictions", False), "Dự đoán đã có khác kết quả mới; cần replace_predictions=true.")
    missing_operational = []
    for rid in missing_models:
        for aspect in ASPECTS:
            existing = old.get((rid, aspect))
            if (rid, aspect) not in manual and (not existing or not existing["label"]):
                # Existing pending rows already have manual labels; scan on apply
                # is authoritative. Here keep complete manual coverage separately.
                if rid in new_ids:
                    missing_operational.append((rid, aspect))
    a8_changes = keyed(rows_file(batch["sources"].get("a8_file")), "đợt nhãn A8")
    for rid, row in a8_changes.items():
        require(rid in master and row.get("review_text_nlp") == master[rid]["review_text_nlp"], "Nhãn A8 dùng mã/nội dung không khớp.")
        require(row.get("A8_label") in {"0", "1"} and row.get("review_status") in {"DA_DUYET", "CAN_XEM_XET"}, "Nhãn/trạng thái A8 không hợp lệ.")
        if row["A8_label"] == "0":
            require(row["review_status"] == "DA_DUYET", "A8=0 phải là quyết định đã duyệt.")
        else:
            codes = set(filter(None, (x.strip() for x in row.get("incident_types", "").split("|"))))
            require(codes and codes <= INCIDENTS, "Loại sự cố A8 thiếu hoặc ngoài bộ mã đã chốt.")
            ev = manual_evidence(master[rid]["review_text_nlp"], "NEG", row.get("A8_evidence", ""))
            if row.get("A8_evidence_spans_json"):
                spans = json.loads(row["A8_evidence_spans_json"])
                check_spans(master[rid]["review_text_nlp"], spans)
                require([s["text"].strip() for s in spans] == [s.strip() for s in row["A8_evidence"].split("||") if s.strip()], "Bằng chứng A8 và vị trí không khớp.")
            else:
                spans = json.loads(ev["evidence_spans_json"])
            require(spans or row["review_status"] == "CAN_XEM_XET", "A8 đã xác nhận cần vị trí bằng chứng rõ ràng.")
            row["A8_evidence_spans_json"] = compact(spans)
    changed_ids = new_ids | {rid for rid, _ in targets} | set(a8_changes)
    affected = sorted({(master[rid]["restaurant_id"], review_period(master[rid])) for rid in changed_ids})
    trend_ids = new_ids | {rid for rid, _ in predictions} | {rid for rid, _ in manual}
    trend_affected = sorted({(master[rid]["restaurant_id"], p) for rid in trend_ids for p in next_periods(review_period(master[rid]))})
    if changing_model:
        trend_affected = sorted({(r["restaurant_id"], p) for r in master.values() for p in next_periods(review_period(r))})
    evidence_profile = manifest.get("evidence_runtime")
    if batch["sources"].get("evidence_profile_file"):
        from src.modeling.predict_evidence_add import evidence_bundle
        new_evidence_profile = evidence_bundle(cfg, jread(batch["sources"]["evidence_profile_file"]))
        require(not evidence_profile or evidence_profile["model_version"] != new_evidence_profile["model_version"]
                or evidence_profile == new_evidence_profile, "Không ghi đè bộ trích khác trọng số/cấu hình vào cùng mã phiên.")
        evidence_profile = new_evidence_profile
    for key, row in evidence.items():
        require(row.get("evidence_source") == "model", "Bằng chứng đầu vào không phải nguồn máy.")
        if evidence_profile:
            require(row.get("evidence_model_version") == evidence_profile["model_version"], "Bằng chứng khác phiên bộ trích đã đăng ký.")
        chosen = predictions.get(key) or old.get(key, {})
        require(row.get("predicted_label") == chosen.get("model_label"), "Bằng chứng không thuộc nhãn máy đã nhập.")
        model_evidence(master[key[0]]["review_text_nlp"], row)
    plan = {"batch_id": batch["batch_id"], "already_applied": False, "new_reviews": len(new_ids),
            "manual_pairs": len(manual), "prediction_pairs_supplied": len(predictions), "evidence_pairs_supplied": len(evidence),
            "a8_decisions_supplied": len(a8_changes), "reviews_after_update": len(master),
            "missing_model_reviews": len(missing_models), "missing_operational_pairs": len(missing_operational),
            "can_apply_without_prediction": not missing_operational and not (changing_model and missing_models),
            "tier2_model_coverage_ready": not missing_models, "tier2_built": False,
            "affected_restaurant_months": affected, "future_tier2_periods_to_recalculate": trend_affected,
            "model_retrained": False, "model_changed": changing_model,
            "requires_full_model_refresh": changing_model, "current_model_version": model_version}
    return {"already_applied": False, "batch": batch, "inputs": inputs, "paths": paths, "manifest": manifest,
            "master": master, "fields": fields, "new_ids": new_ids, "manual": manual,
            "predictions": predictions, "evidence": evidence, "old": old, "a8_changes": a8_changes,
            "training_ids": training_ids, "excluded_ids": excluded_ids, "base_manifest": base_manifest,
            "missing_models": missing_models, "model_version": model_version, "plan": plan,
            "runtime_model": profile, "changing_model": changing_model, "evidence_runtime": evidence_profile}


@contextmanager
def update_lock(base):
    base.mkdir(parents=True, exist_ok=True)
    lock = base / ".cap_nhat_system.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("Đang có lần cập nhật khác hoặc lần trước dừng đột ngột. Kiểm tra tiến trình trước khi gỡ khóa.") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(compact({"pid": os.getpid(), "time": now()}))
        yield
    finally:
        lock.unlink(missing_ok=True)


def apply_batch(cfg, batch_path, run_predictions=False, device="auto", predictor=None,
                run_evidence=False, evidence_predictor=None, build_alerts=True):
    from src.pipeline.prepare_system_data import load_system_inputs
    migrate_legacy_outputs(cfg)
    base = location(cfg, "system_data_dir")
    output = system_output_root(cfg)
    tier2_config = cfg["_root"] / "configs/alerts/tang2/trend_alerts.json"
    tier2_enabled = tier2_config.is_file() and jread(tier2_config).get("enabled") is True
    with update_lock(output):
        work = prepare_batch(cfg, batch_path)
        if work["already_applied"]:
            if build_alerts:
                # System không nhập lặp; cảnh báo tự kiểm tra chữ ký và chỉ tạo
                # lại khi mã/cấu hình/kết quả đã đổi, kể cả sau khi chuyển nơi lưu.
                from src.alerts.build_content_alerts import run as build_content
                build_content(cfg, use_system=True)
                if tier2_enabled:
                    from src.alerts.tang2.build_trend_alerts import run as build_trend
                    build_trend(cfg)
            return {"status": "already_applied", "batch_id": work["batch"]["batch_id"], "summary": work["manifest"]["summary"],
                    "system_directory": str(current_paths(cfg)["directory"])}
        batch, master = work["batch"], work["master"]
        if run_predictions and work["missing_models"]:
            if predictor is None:
                from src.modeling.predict_system_updates import predict_reviews
                predictor = predict_reviews
            jobs = cfg["_root"] / "data/predictions/system_updates" / batch["batch_id"]
            jobs.mkdir(parents=True, exist_ok=True)
            records = [master[rid] for rid in sorted(work["missing_models"])]
            predictions_path = jobs / "aspect_predictions.csv"
            from src.modeling.runtime_model import prediction_config
            predictor(prediction_config(cfg, work["runtime_model"]), records, predictions_path, device=device)
            generated = prediction_changes(rows_file(predictions_path), master, work["model_version"])
            require({rid for rid, _ in generated} == work["missing_models"], "Bộ dự đoán chưa trả đủ hoặc trả thừa nhận xét được yêu cầu.")
            work["predictions"].update(generated)
            work["plan"]["predicted_reviews_this_run"] = len(records)
            work["generated_predictions_path"] = predictions_path
        require(not work["plan"]["missing_operational_pairs"] or run_predictions,
                "Nhận xét mới chưa có đủ nhãn tay hoặc dự đoán. Thêm predictions_file hoặc chạy --predict cùng --apply.")
        if work["changing_model"]:
            require({rid for rid, _ in work["predictions"]} == set(master),
                    "Thay mô hình cần đủ dự đoán mới trên toàn bộ corpus; không kích hoạt bản trộn mô hình.")
        if run_evidence:
            require(work["evidence_runtime"], "Chạy bằng chứng cần evidence_profile_file cho lần đầu.")
            if evidence_predictor is None:
                from src.modeling.predict_evidence_add import predict_evidence
                evidence_predictor = predict_evidence
            # Không lấy nhãn tay làm đầu vào giả cho bộ trích bằng chứng máy.
            machine = {(r["review_id"], r["aspect"]): r for r in csv_rows(work["paths"]["aspects"])}
            for rid in work["new_ids"]:
                for a in ASPECTS:
                    machine[(rid, a)] = empty_aspect(master[rid], a)
            selected = []
            for key, row in sorted(machine.items()):
                change = work["predictions"].get(key, row)
                require(change.get("model_label"), "Thiếu nhãn máy để dự đoán bằng chứng.")
                if key in work["evidence"]:
                    continue
                if key in work["predictions"] or row.get("model_evidence_version", row.get("evidence_model_version")) != work["evidence_runtime"]["model_version"]:
                    selected.append({"review_id": key[0], "aspect": key[1],
                                     "predicted_label": change["model_label"], "model_version": work["model_version"]})
            if selected:
                jobs = cfg["_root"] / "data/predictions/system_updates" / batch["batch_id"]
                jobs.mkdir(parents=True, exist_ok=True)
                target = jobs / "evidence_predictions.csv"
                evidence_predictor(cfg, work["evidence_runtime"], master, selected, target, device=device)
                generated_ev = {(r["review_id"], r["aspect"]): r for r in rows_file(target)}
                require(len(generated_ev) == len(selected) and set(generated_ev) == {(r["review_id"], r["aspect"]) for r in selected},
                        "Bộ trích bằng chứng trả thiếu/thừa cặp.")
                require(all(r.get("model_version") == work["model_version"] for r in generated_ev.values()), "Bằng chứng khác phiên nhãn.")
                work["evidence"].update(generated_ev)
                work["generated_evidence_path"] = target
            work["plan"]["evidence_pairs_predicted_this_run"] = len(selected)
        updates = output
        updates.mkdir(parents=True, exist_ok=True)
        destination = updates / batch["batch_id"]
        require(not destination.exists(), "Đã có thư mục đợt chưa kích hoạt. Giữ để kiểm tra và dùng batch_id mới.")
        staging = Path(tempfile.mkdtemp(prefix="building_", dir=updates))
        counts, manual_ids, full_manual, pending_models = Counter(), set(), Counter(), Counter()
        activated = False
        published_system = False
        owned_alert_dirs = []
        try:
            def merged_rows():
                source = csv_rows(work["paths"]["aspects"])
                added = (empty_aspect(master[rid], a) for rid in sorted(work["new_ids"]) for a in ASPECTS)
                for original in heapq.merge(source, added, key=lambda r: (r["review_id"], r["aspect"])):
                    row = {k: original.get(k, "") for k in ASPECT_FIELDS}
                    rid, aspect = row["review_id"], row["aspect"]
                    key, text = (rid, aspect), master[rid]["review_text_nlp"]
                    retain_source_evidence(row, text)
                    if (work["evidence_runtime"] and row.get("model_evidence_version")
                            and row["model_evidence_version"] != work["evidence_runtime"]["model_version"]):
                        clear_model_evidence(row, text)
                    prediction = work["predictions"].get(key)
                    if prediction:
                        old_label = row["model_label"]
                        row.update({k: str(v) for k, v in prediction.items()})
                        if work["changing_model"] or old_label != row["model_label"] or row.get("model_evidence_source") in {"", "none"}:
                            clear_model_evidence(row, text)
                    if key in work["evidence"]:
                        evidence = work["evidence"][key]
                        require(evidence.get("predicted_label") == row["model_label"], "Bằng chứng không thuộc nhãn máy hiện hành.")
                        row.update(tagged_evidence(model_evidence(text, evidence), "model"))
                    if key in work["manual"]:
                        row.update(work["manual"][key])
                    select_operational_source(row, text)
                    require(row["label"] in LABELS, "Thiếu nhãn sử dụng cho " + str(key))
                    if row["label_source"] == "manual":
                        full_manual[rid] += 1
                    if row.get("manual_label") in LABELS:manual_ids.add(rid)
                    if not row["model_label"]:
                        pending_models[rid] += 1
                    row.update(used_for_training=str(rid in work["training_ids"]).lower(),
                               excluded_vlsp_overlap=str(rid in work["excluded_ids"]).lower())
                    counts["pairs"] += 1
                    counts["manual_pairs" if row["label_source"] == "manual" else "model_pairs"] += 1
                    counts["pending_evidence_pairs"] += int(row["label"] != "NONE" and not flag(row["evidence_verified"]))
                    counts["stored_model_evidence_pairs"] += int(row.get("model_evidence_source") == "model")
                    counts["stored_manual_evidence_pairs"] += int(row.get("manual_evidence_source") == "manual")
                    counts["pending_model_evidence_pairs"] += int(row["model_label"] != "NONE" and row.get("model_evidence_source") != "model")
                    yield row
            # has_manual_label is per-review, while label_source remains per pair.
            temp_aspects = staging / "aspects_pass1.csv"
            write_csv(temp_aspects, ASPECT_FIELDS, merged_rows())
            def final_aspects():
                for row in csv_rows(temp_aspects):
                    row["has_manual_label"] = str(row["review_id"] in manual_ids).lower()
                    yield row
            write_csv(staging / cfg["system_aspects_file"], ASPECT_FIELDS, final_aspects())
            temp_aspects.unlink()
            for rid, row in master.items():
                row["has_manual_label"] = str(rid in manual_ids).lower()
                row["used_for_training"] = str(rid in work["training_ids"]).lower()
                row["excluded_vlsp_overlap"] = str(rid in work["excluded_ids"]).lower()
            write_csv(staging / "reviews_master.csv", work["fields"], (master[rid] for rid in sorted(master)))
            positives = {r["review_id"]: r for r in work["inputs"]["a8"]}
            negatives = set(work["manifest"]["a8_known_negative_review_ids"])
            for rid, change in work["a8_changes"].items():
                positives.pop(rid, None); negatives.discard(rid)
                if change["A8_label"] == "0":
                    negatives.add(rid); continue
                row = {key: "" for key in A8_FIELDS}
                row.update(change)
                row.update({k: master[rid][k] for k in ("review_id", "restaurant_id", "restaurant_name", "review_date_iso", "city_std", "text_group_id", "review_text_nlp")})
                row.update(label_source="NHAN_DA_CHOT", A8_alert_title="A8 — " + ("ĐÃ XÁC NHẬN" if row["review_status"] == "DA_DUYET" else "CẦN XEM XÉT"),
                           A8_evidence_source="NHAN_TAY_DA_XAC_NHAN" if row["review_status"] == "DA_DUYET" else "NHAN_TAY_CAN_XEM_XET")
                positives[rid] = row
            write_csv(staging / cfg["system_a8_file"], A8_FIELDS, (positives[rid] for rid in sorted(positives)))
            copied = []
            (staging / "inputs").mkdir()
            (staging / "inputs/batch.json").write_bytes(batch["path"].read_bytes())
            copied.append("inputs/batch.json")
            for key, source in batch["sources"].items():
                target = staging / "inputs" / (key + source.suffix)
                shutil.copy2(source, target)
                require(sha(target) == batch["hashes"][key] == sha(source), "Nguồn đã đổi khi đang cập nhật.")
                copied.append("inputs/" + target.name)
            if work.get("generated_predictions_path"):
                shutil.copy2(work["generated_predictions_path"], staging / "inputs/generated_predictions.csv")
                copied.append("inputs/generated_predictions.csv")
            if work.get("generated_evidence_path"):
                shutil.copy2(work["generated_evidence_path"], staging / "inputs/generated_evidence.csv")
                copied.append("inputs/generated_evidence.csv")
            require(counts["pairs"] == len(master) * 7, "System cập nhật thiếu/thừa cặp khía cạnh.")
            summary = {"reviews": len(master), "restaurants": len({r["restaurant_id"] for r in master.values()}),
                       "manual_reviews": len(manual_ids), "fully_manual_reviews": sum(n == 7 for n in full_manual.values()),
                       "model_only_reviews": len(master) - len(manual_ids), "aspect_pairs": counts["pairs"],
                       "manual_aspect_pairs": counts["manual_pairs"], "model_aspect_pairs": counts["model_pairs"],
                       "pending_model_reviews": len(pending_models), "pending_evidence_pairs": counts["pending_evidence_pairs"],
                       "stored_model_evidence_pairs": counts["stored_model_evidence_pairs"],
                       "stored_manual_evidence_pairs": counts["stored_manual_evidence_pairs"],
                       "pending_model_evidence_pairs": counts["pending_model_evidence_pairs"],
                       "a8_alerts": len(positives), "a8_by_status": dict(Counter(r["review_status"] for r in positives.values())),
                       "a8_explicit_negative_reviews": len(negatives), "a8_unlabeled_reviews": len(master) - len(positives) - len(negatives),
                       "base_files_modified": False, "model_retrained": False,
                       "alerts_built_during_update": build_alerts, "tier2_built": tier2_enabled and build_alerts}
            plan = dict(work["plan"], missing_model_reviews=len(pending_models), tier2_model_coverage_ready=not pending_models)
            if work.get("generated_predictions_path"):
                model_ids = {r["review_id"] for r in rows_file(work["generated_predictions_path"])}
                affected = set(map(tuple, plan["future_tier2_periods_to_recalculate"]))
                affected.update((master[rid]["restaurant_id"], p) for rid in model_ids for p in next_periods(review_period(master[rid])))
                plan["future_tier2_periods_to_recalculate"] = sorted(affected)
            atomic_json(staging / "update_plan.json", plan)
            all_affected = set(map(tuple, plan["future_tier2_periods_to_recalculate"]))
            for earlier in work["manifest"].get("update_history", []):
                all_affected.update(map(tuple, earlier["plan"]["future_tier2_periods_to_recalculate"]))
            contract = {"tier2_built": tier2_enabled and build_alerts, "model_coverage_ready": not pending_models,
                        "pending_model_review_ids": sorted(pending_models), "label_column": "label",
                        "label_selection": "confirmed_manual_else_model",
                        "evaluation_excluded_review_ids": sorted(work["excluded_ids"]),
                        "model_version": work["model_version"], "period": "month", "timezone": "Asia/Ho_Chi_Minh",
                        "metric": "NEG / all_reviews", "missing_predictions_are_not_NONE": True,
                        "baseline_windows": [2, 3], "recalculation_mode": "full_current_snapshot", "configuration_source": "configs/alerts/tang2/trend_alerts.json" if tier2_enabled else "alert_policy.trend",
                        "coverage_configuration_source": "configs/alerts/tang2/trend_alerts.json" if tier2_enabled else None,
                        "periods_to_recalculate": sorted(all_affected),
                        "latest_batch_periods_to_recalculate": plan["future_tier2_periods_to_recalculate"],
                        "consumer_must_track_last_processed_batch": True}
            atomic_json(staging / "tier2_input_contract.json", contract)
            entry = {"batch_id": batch["batch_id"], "identity": batch["identity"], "applied_at": now(),
                     "annotation_version": batch["definition"].get("annotation_version", "unspecified"),
                     "source_hashes": batch["hashes"], "plan": plan}
            files = ["reviews_master.csv", cfg["system_aspects_file"], cfg["system_a8_file"],
                     "update_plan.json", "tier2_input_contract.json", *copied]
            _, _, base_manifest = prepared_data(cfg)
            manifest = {"version": VERSION, "schema_version": 2, "status": "complete", "created_at_utc": now(),
                        "update_mode": "all" if build_alerts else "system_only",
                        "output_layout_version": LAYOUT_VERSION,
                        "source_evidence_schema_version": SOURCE_EVIDENCE_SCHEMA,
                        "signature": {"version": VERSION, "config_sha256": sha(cfg["_config"]),
                                      "batch_identity": batch["identity"], "previous_manifest_sha256": sha(work["paths"]["manifest"])},
                        "base_identity": {"master_sha256": sha(base / "reviews_master.csv"), "train_sha256": sha(base / "train_reviews.csv"),
                                          "training_dataset_sha256": base_manifest["training_dataset_sha256"]},
                        "summary": summary, "current_model_version": work["model_version"],
                        "runtime_model": work["runtime_model"], "evidence_runtime": work["evidence_runtime"],
                        "a8_known_negative_review_ids": sorted(negatives), "operational_label_column": "label", "trend_label_column": "label",
                        "label_selection": "confirmed_manual_else_model",
                        "update_history": [*work["manifest"].get("update_history", []), entry],
                        "previous_batch_id": work["manifest"].get("update_history", [{}])[-1].get("batch_id"),
                        "output_sha256": {name: sha(staging / name) for name in files}}
            atomic_json(staging / cfg["system_inputs_manifest"], manifest)
            staged_cfg = dict(cfg, _system_snapshot=staging)
            verify_snapshot(staged_cfg, current_paths(staged_cfg))
            # Đặt System đúng nơi trước khi tạo cảnh báo, nhưng chưa kích hoạt.
            final_cfg = dict(cfg, _system_snapshot=destination, _output_batch_id=batch["batch_id"])
            if build_alerts:
                for tier in ([1, 2] if tier2_enabled else [1]):
                    target = alert_output_dir(final_cfg, tier)
                    require(not target.exists(), "Đã có kết quả cảnh báo của mã đợt này; dùng mã đợt mới: " + str(target))
                    owned_alert_dirs.append(target)
            staging.rename(destination)
            published_system = True
            if build_alerts:
                from src.alerts.build_content_alerts import run as build_content
                staged_inputs = load_system_inputs(cfg["_root"], config=cfg["_config"], snapshot_dir=destination)
                build_content(final_cfg, inputs_override=staged_inputs, use_system=True)
                if tier2_enabled:
                    from src.alerts.tang2.build_trend_alerts import run as build_trend
                    trend_inputs = dict(staged_inputs, aspects=csv_rows(destination / cfg["system_aspects_file"]))
                    build_trend(final_cfg, inputs_override=trend_inputs)
            # Mặc định chờ cả hai tầng; --system-only chỉ kích hoạt dữ liệu đã kiểm tra.
            require(sha(work["paths"]["manifest"]) == manifest["signature"]["previous_manifest_sha256"], "System đang dùng đã đổi trong lúc cập nhật.")
            require(read_batch(batch_path)["identity"] == batch["identity"], "Định nghĩa đợt hoặc đầu vào đã đổi khi đang cập nhật.")
            if work["runtime_model"] and (work["changing_model"] or work.get("generated_predictions_path")):
                from src.modeling.runtime_model import model_bundle
                model_bundle(cfg, work["runtime_model"])
            if run_evidence and work["evidence_runtime"]:
                from src.modeling.predict_evidence_add import evidence_bundle
                evidence_bundle(cfg, work["evidence_runtime"])
            atomic_json(output / POINTER, {"batch_id": batch["batch_id"], "manifest_sha256": sha(destination / cfg["system_inputs_manifest"]),
                                           "output_layout_version": LAYOUT_VERSION, "activated_at": now()})
            activated = True
            return {"status": "applied", "batch_id": batch["batch_id"], "summary": summary,
                    "system_directory": str(destination), "plan": plan}
        finally:
            if not activated:
                for target in owned_alert_dirs:
                    shutil.rmtree(target, ignore_errors=True)
                if published_system:
                    shutil.rmtree(destination, ignore_errors=True)
            if staging.exists():
                shutil.rmtree(staging)


def _copy_file(source, target, expected):
    require(source.is_file() and sha(source) == expected, "Tệp cũ thiếu/đã đổi: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(target.is_file() and sha(target) == expected, "Đường dẫn mới có nội dung khác; chưa ghi đè: " + str(target))
    else:
        shutil.copy2(source, target)
    require(sha(source) == expected and sha(target) == expected, "Tệp đổi trong lúc chuyển: " + str(source))

def _system_files(source):
    for path in source.rglob("*"):
        require(not path.is_symlink(), "Không chuyển liên kết tượng trưng: " + str(path))
        relative = path.relative_to(source)
        if relative.parts[0] != "alerts" and path.is_file():
            yield relative

def _copy_system(source, target, cfg):
    old_cfg = dict(cfg, _allow_config_change=True)
    fields = {"directory": source, "manifest": source / cfg["system_inputs_manifest"],
              "aspects": source / cfg["system_aspects_file"]}
    verify_snapshot(old_cfg, fields)
    if not target.exists():
        stage = Path(tempfile.mkdtemp(prefix=".dang_chuyen_", dir=target.parent))
        try:
            for relative in _system_files(source):
                path = source / relative
                _copy_file(path, stage / relative, sha(path))
            stage.rename(target)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    for relative in _system_files(source):
        require((target / relative).is_file() and sha(target / relative) == sha(source / relative),
                "Bản System ở reports khác bản cũ; chưa xóa nguồn: " + str(relative))
    verify_snapshot(old_cfg, {"directory": target, "manifest": target / cfg["system_inputs_manifest"],
                             "aspects": target / cfg["system_aspects_file"]})

def _tier1(cfg, source_data, source_report, source_manifest, batch):
    if not source_manifest.is_file():
        require(not source_data.exists(), "Có CSV tầng 1 nhưng thiếu hồ sơ; chưa chuyển: " + str(source_data))
        return False
    manifest = jread(source_manifest)
    require(manifest.get("status") == "complete", "Tầng 1 cũ chưa hoàn tất.")
    target = alert_output_root(cfg, 1) / batch
    manifest_path = target / "content_run_manifest.json"
    if manifest_path.exists():
        require(jread(manifest_path)["signature"] == manifest["signature"], "Tầng 1 ở reports thuộc lần khác; chưa chuyển đè.")
    _copy_file(source_data, target / "content_alerts.csv", manifest["output_sha256"]["data"])
    _copy_file(source_report, target / "kiem_tra_tang_1.md", manifest["output_sha256"]["report"])
    manifest = deepcopy(manifest)
    manifest["output_layout_version"] = LAYOUT_VERSION
    manifest["output_paths"] = {key: (target / name).relative_to(cfg["_root"]).as_posix()
                               for key, name in {"data": "content_alerts.csv", "report": "kiem_tra_tang_1.md",
                                                 "manifest": "content_run_manifest.json"}.items()}
    atomic_json(manifest_path, manifest)
    return True

def _tier2(cfg, source, batch, cleanup_reports, preserve_unrecognized=False):
    if not source.exists():
        return 0
    def unrecognized(reason):
        if not preserve_unrecognized:
            raise ValueError(reason)
        print("Giữ nguyên kết quả tầng 2 cũ tại " + str(source) + ": " + reason
              + " System tiếp tục cập nhật; tạo tầng 2 mới bằng notebook.", flush=True)
        return 0
    require(not (source / ".building.lock").exists(), "Tầng 2 cũ đang chạy hoặc còn khóa; chưa chuyển.")
    extras = sorted(p.name for p in source.iterdir() if p.name not in {"current_trend.json", "lan_chay"})
    if extras:
        return unrecognized("Có tệp/thư mục ngoài bố cục được hỗ trợ: " + ", ".join(extras[:8]))
    pointer_file = source / "current_trend.json"
    if not pointer_file.is_file():
        return unrecognized("Thiếu current_trend.json; chưa chuyển hoặc xóa nguồn.")
    pointer = jread(pointer_file)
    if pointer.get("layout_version") != 2 or not (source / "lan_chay").is_dir():
        return unrecognized("Bố cục tầng 2 cũ chưa được hỗ trợ tự chuyển.")
    # Kiểm tra cả cây trước khi chép: nếu phải giữ nguồn thì chưa dọn báo cáo nào.
    runs = []
    for run in sorted((source / "lan_chay").iterdir()):
        if not run.is_dir() or not (run / "trend_manifest.json").is_file():
            return unrecognized("Có lần chạy chưa hoàn tất hoặc chưa nhận dạng: " + run.name)
        manifest = jread(run / "trend_manifest.json")
        if manifest.get("status") != "complete" or manifest.get("layout_version") != 2:
            return unrecognized("Lần chạy chưa hoàn tất hoặc khác bố cục hỗ trợ: " + run.name)
        declared = {"trend_manifest.json", *manifest["output_sha256"]}
        if not all(p.is_file() and p.name in declared for p in run.iterdir()):
            return unrecognized("Lần chạy chứa tệp ngoài hồ sơ: " + run.name)
        runs.append((run, manifest))
    current = safe_name(pointer["directory_name"])
    require(sha(source / "lan_chay" / current / "trend_manifest.json") == pointer["manifest_sha256"],
            "Con trỏ tầng 2 cũ không khớp.")
    target_root = alert_output_root(cfg, 2) / batch
    total = 0
    for run, manifest in runs:
        name = safe_name(run.name)
        relative = Path(manifest["report_directory"])
        reports = (cfg["_root"] / relative).resolve()
        require(not relative.is_absolute() and ".." not in relative.parts
                and reports.is_relative_to(alert_output_root(cfg, 2)), "Báo cáo tầng 2 cũ nằm ngoài nhánh hợp lệ.")
        target = target_root / "lan_chay" / name
        if (target / "trend_manifest.json").exists():
            require(jread(target / "trend_manifest.json")["signature"] == manifest["signature"],
                    "Đã có lần chạy khác ở đường dẫn tầng 2 mới; chưa chuyển đè.")
        for file, digest in manifest["output_sha256"].items():
            require(Path(file).name == file, "Tên tệp tầng 2 không hợp lệ.")
            _copy_file(run / file, target / file, digest)
        for file, digest in manifest["report_sha256"].items():
            require(Path(file).name == file, "Tên báo cáo tầng 2 không hợp lệ.")
            _copy_file(reports / file, target / file, digest)
            cleanup_reports.append((reports / file, target / file, digest))
        updated = dict(manifest, layout_version=LAYOUT_VERSION,
                       report_directory=target.relative_to(cfg["_root"]).as_posix())
        atomic_json(target / "trend_manifest.json", updated)
        total += 1
    new_pointer = dict(pointer, layout_version=LAYOUT_VERSION,
                       manifest_sha256=sha(target_root / "lan_chay" / current / "trend_manifest.json"))
    pointer_target = target_root / "current_trend.json"
    if pointer_target.is_file():
        require(jread(pointer_target).get("run_id") == new_pointer.get("run_id"),
                "Tầng 2 ở reports đã chọn lần khác; chưa thay con trỏ.")
    atomic_json(pointer_target, new_pointer)
    return total

def migrate_legacy_outputs(cfg):
    """Kiểm tra, chép đủ, đổi con trỏ rồi mới dọn các đầu ra cũ trong data."""
    base = location(cfg, "system_data_dir")
    old_root = base / "updates"
    old_dirs = sorted(p for p in old_root.iterdir() if p.is_dir()) if old_root.is_dir() else []
    legacy_alerts = cfg["_root"] / "data/alerts/foody_v01"
    base_manifest = cfg["_root"] / "reports/alerts/foody_v01/content_run_manifest.json"
    old_pointer = base / POINTER
    if not old_dirs and not old_pointer.exists() and not legacy_alerts.exists() and not base_manifest.exists():
        return {"system_batches_moved": 0, "tier1_batches_moved": 0, "tier2_runs_moved": 0}
    output = system_output_root(cfg)
    counts = {"system_batches_moved": 0, "tier1_batches_moved": 0, "tier2_runs_moved": 0}
    with update_lock(base), update_lock(output):
        cleanup_reports = []
        cleanup_base_files = []
        base_tier2_moved = False
        for source in old_dirs:
            batch = safe_name(source.name)
            require((source / cfg["system_inputs_manifest"]).is_file(), "Có đợt System cũ chưa hoàn tất; chưa chuyển: " + str(source))
            _copy_system(source, output / batch, cfg)
            alerts = source / "alerts"
            if alerts.exists():
                require(all(p.name in {"content_alerts.csv", "kiem_tra_tang_1.md", "content_run_manifest.json", "tang2"}
                            for p in alerts.iterdir()), "Có tệp cảnh báo chưa nhận dạng trong System cũ; chưa xóa nguồn.")
                counts["tier1_batches_moved"] += int(_tier1(cfg, alerts / "content_alerts.csv",
                    alerts / "kiem_tra_tang_1.md", alerts / "content_run_manifest.json", batch))
                counts["tier2_runs_moved"] += _tier2(cfg, alerts / "tang2", batch, cleanup_reports)
            counts["system_batches_moved"] += 1
        if base_manifest.exists():
            report = base_manifest.with_name("kiem_tra_tang_1.md")
            data = legacy_alerts / "content_alerts.csv"
            _tier1(cfg, data, report, base_manifest, "ban_dau")
            counts["tier1_batches_moved"] += 1
            cleanup_base_files.extend([data, report, base_manifest])
        if (legacy_alerts / "tang2").exists():
            moved = _tier2(cfg, legacy_alerts / "tang2", "ban_dau", cleanup_reports,
                           preserve_unrecognized=True)
            counts["tier2_runs_moved"] += moved
            base_tier2_moved = moved > 0
        if old_pointer.is_file():
            pointer = jread(old_pointer)
            target = output / safe_name(pointer["batch_id"])
            require(sha(target / cfg["system_inputs_manifest"]) == pointer["manifest_sha256"], "Bản được chuyển khác con trỏ System cũ.")
            replace_pointer = not (output / POINTER).exists()
            if not replace_pointer:
                # Có thể đang tiếp tục một lần chuyển dở. Chỉ chọn bản kế thừa
                # đầy đủ lịch sử; không âm thầm bỏ một quyết định nhãn mới hơn.
                new_paths = current_paths(cfg)
                new_manifest = jread(new_paths["manifest"])
                old_manifest = jread(target / cfg["system_inputs_manifest"])
                new_history = {(r["batch_id"], r["identity"]) for r in new_manifest["update_history"]}
                old_history = {(r["batch_id"], r["identity"]) for r in old_manifest["update_history"]}
                old_last = old_manifest["update_history"][-1]
                new_last = new_manifest["update_history"][-1]
                if (old_last["batch_id"], old_last["identity"]) not in new_history:
                    require((new_last["batch_id"], new_last["identity"]) in old_history,
                            "Hai con trỏ System thuộc hai nhánh khác nhau; giữ nguồn để đối chiếu, chưa chọn thay.")
                    replace_pointer = True
            if replace_pointer:
                atomic_json(output / POINTER, dict(pointer, output_layout_version=LAYOUT_VERSION, moved_at=now()))
        if (output / POINTER).is_file():
            verify_snapshot(dict(cfg, _allow_config_change=True), current_paths(cfg))
        require(not old_dirs or (output / POINTER).is_file(), "Chưa có bản System hoạt động ở reports; giữ nguồn cũ.")
        # Chỉ dọn sau khi toàn bộ bản sao và con trỏ đã được kiểm tra.
        for old, new, digest in cleanup_reports:
            require(sha(old) == digest and sha(new) == digest, "Báo cáo đổi trong lúc chuyển; giữ nguồn.")
        for source in old_dirs:
            for relative in _system_files(source):
                require(sha(source / relative) == sha(output / source.name / relative), "System đổi trong lúc chuyển; giữ nguồn.")
        for source in old_dirs:
            shutil.rmtree(source)
        old_pointer.unlink(missing_ok=True)
        for old, new, digest in cleanup_reports:
            old.unlink(missing_ok=True)
        for file in cleanup_base_files:
            file.unlink(missing_ok=True)
        if base_tier2_moved:
            shutil.rmtree(legacy_alerts / "tang2")
        # Chỉ bỏ thư mục rỗng; không xóa tệp ngoài phạm vi đã nhận dạng.
        stop = alert_output_root(cfg, 2)
        for old, _, _ in cleanup_reports:
            parent = old.parent
            while parent != stop and parent.is_relative_to(stop):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        for parent in [old_root, legacy_alerts]:
            try:
                parent.rmdir()
            except OSError:
                pass
    if any(counts.values()):
        print("Đã chuyển đầu ra cũ sang reports:", json.dumps(counts, ensure_ascii=False), flush=True)
    return counts


def console_result(result):
    """Chỉ in thống kê; danh sách tháng đầy đủ đã lưu trong update_plan.json."""
    output = dict(result)
    plan = output.pop("plan", output)
    for key in ("affected_restaurant_months", "future_tier2_periods_to_recalculate"):
        if key in plan:
            output[key + "_count"] = len(plan[key])
            output.pop(key, None)
    if output.get("system_directory"):
        output["update_plan_file"] = str(Path(output["system_directory"]) / "update_plan.json")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Nhập và kích hoạt sau khi kiểm tra cả system và tầng 1.")
    parser.add_argument("--system-only", action="store_true", help="Chỉ cập nhật System, chưa tạo hai tầng cảnh báo.")
    parser.add_argument("--predict", action="store_true", help="Dùng mô hình hiện tại dự đoán phần còn thiếu; không huấn luyện.")
    parser.add_argument("--predict-evidence", action="store_true", help="Chạy mô hình bằng chứng đã đăng ký; không huấn luyện.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    require(not args.system_only or args.apply, "--system-only cần dùng cùng --apply.")
    cfg = load_settings(args.config, args.project_root)
    if args.apply:
        result = apply_batch(cfg, args.batch, args.predict, args.device, run_evidence=args.predict_evidence,
                             build_alerts=not args.system_only)
    else:
        work = prepare_batch(cfg, args.batch)
        result = {"status": "already_applied", "batch_id": work["batch"]["batch_id"]} if work["already_applied"] else work["plan"]
    print(json.dumps(console_result(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, OSError, json.JSONDecodeError) as error:
        print("DỪNG:", error)
        raise SystemExit(2)
