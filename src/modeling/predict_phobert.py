"""Dự đoán nhãn máy từ nền foody_v01; lưu từng phần và tiếp tục an toàn.

Đây là dự đoán gốc, không ghi đè nhãn tay/bằng chứng vận hành. Sau khi có
kết quả bằng chứng tương ứng, cập nhật system bằng src.pipeline.prepare_system_data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from .phobert_system_common import (
    ASPECTS, LABELS, PairCollator, Segmenter, aggregate_logits, atomic_csv, atomic_json,
    choose_device, encode_pair, environment, load_settings, location, now, object_hash,
    prepared_data, read_csv, read_json, segmenter_files, sha256_file,
)

COLUMNS = ["review_id", "aspect", "predicted_label"] + [f"p_{label}" for label in LABELS] + [
    "confidence", "margin", "model_version", "label_source", "has_manual_label", "used_for_training",
    "excluded_vlsp_overlap", "window_count"]


class SegmentationCache:
    def __init__(self, path, signature):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS texts (signature TEXT NOT NULL, text_hash TEXT NOT NULL, segmented TEXT NOT NULL, PRIMARY KEY(signature,text_hash))")
        self.signature = signature

    def get(self, text):
        key = object_hash(str(text))
        row = self.db.execute("SELECT segmented FROM texts WHERE signature=? AND text_hash=?",
                              (self.signature, key)).fetchone()
        return row[0] if row else None

    def put(self, text, segmented):
        self.db.execute("INSERT OR REPLACE INTO texts VALUES (?,?,?)",
                        (self.signature, object_hash(str(text)), str(segmented)))

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()


def validate_part(df, review_ids, model_version, training_ids):
    if list(df.columns) != COLUMNS or len(df) != len(review_ids) * 7:
        raise ValueError("Phần dự đoán sai cấu trúc/số dòng.")
    expected = {(str(rid), aspect) for rid in review_ids for aspect in ASPECTS}
    actual = set(zip(df.review_id, df.aspect))
    if len(actual) != len(df) or actual != expected:
        raise ValueError("Phần dự đoán thiếu/trùng cặp nhận xét–khía cạnh.")
    p = df[[f"p_{label}" for label in LABELS]].astype(float).to_numpy()
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Xác suất không hợp lệ.")
    if not df.predicted_label.eq(np.asarray(LABELS)[p.argmax(axis=1)]).all():
        raise ValueError("Nhãn dự đoán không khớp xác suất.")
    if not np.allclose(df.confidence.astype(float), p.max(axis=1), atol=1e-6):
        raise ValueError("confidence không khớp xác suất.")
    if not df.model_version.eq(model_version).all() or not df.label_source.eq("model").all():
        raise ValueError("Phiên bản/nguồn nhãn dự đoán không khớp.")
    if not df.used_for_training.str.lower().eq(df.review_id.isin(training_ids).map({True: "true", False: "false"})).all():
        raise ValueError("Cờ used_for_training không khớp mô hình.")


def model_metadata(cfg):
    model_dir = location(cfg, "model_dir")
    complete = read_json(model_dir / "training_complete.json")
    if complete.get("training_completed") is not True:
        raise ValueError("Mô hình chưa huấn luyện hoàn thành.")
    for name, expected in complete["files_sha256"].items():
        if not (model_dir / name).exists() or sha256_file(model_dir / name) != expected:
            raise ValueError(f"Tệp mô hình thiếu/đã đổi: {name}")
    absa = read_json(model_dir / "absa_config.json")
    prep = read_json(model_dir / "preprocessing.json")
    if absa["labels"] != LABELS or absa["aspects"] != ASPECTS or absa["pooling"] != "mean_logits_over_all_review_windows":
        raise ValueError("Mô hình không đúng schema/gộp đoạn.")
    if absa["model_version"] != cfg["model_version"]:
        raise ValueError("Phiên bản mô hình trong cấu hình không khớp.")
    ids = read_csv(model_dir / "training_review_ids.csv")
    if ids.review_id.duplicated().any():
        raise ValueError("ID huấn luyện lưu trong mô hình bị trùng.")
    return model_dir, complete, absa, prep, set(ids.review_id)


def seed_cache(cache, cfg, prep, master):
    path = location(cfg, "segmented_file")
    if not path.exists() or sha256_file(path) != prep["training_segmented_file_sha256"]:
        return 0
    old = read_csv(path)
    if not {"review_id", "review_text_nlp", "phobert_text"}.issubset(old.columns):
        return 0
    current = master.set_index("review_id").review_text_nlp
    count = 0
    for row in old.itertuples(index=False):
        if row.review_id in current and current[row.review_id] == row.review_text_nlp:
            cache.put(row.review_text_nlp, row.phobert_text)
            count += 1
    cache.commit()
    return count


def merge_parts(out, manifest, master, training_ids):
    target = out / "aspect_predictions.csv"
    temp = out / "aspect_predictions.csv.tmp"
    chunk_size = manifest["identity_data"]["review_chunk_size"]
    count = 0
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for part in manifest["parts"]:
                path = out / part["file"]
                if sha256_file(path) != part["sha256"]:
                    raise ValueError("Phần dự đoán đã bị thay đổi: " + str(path))
                data = read_csv(path)
                start = part["index"] * chunk_size
                ids = master.iloc[start:start + chunk_size].review_id.tolist()
                validate_part(data, ids, manifest["identity_data"]["model_version"], training_ids)
                writer.writerows(data.to_dict("records"))
                count += len(data)
        if count != len(master) * 7:
            raise ValueError("Tổng số dự đoán chưa đủ; không đánh dấu hoàn thành.")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return count, sha256_file(target)


def run(cfg, args):
    train, master, data_manifest = prepared_data(cfg)
    model_dir, completion, absa, prep, training_ids = model_metadata(cfg)
    if training_ids != set(train.review_id) or absa["training_dataset_sha256"] != data_manifest["training_dataset_sha256"]:
        raise ValueError("Mô hình không học trên train_reviews.csv của hệ thống này.")
    if args.scope == "unlabeled":
        master = master.loc[master.has_manual_label.str.lower().eq("false")].reset_index(drop=True)
    if args.limit:
        master = master.head(args.limit).copy()
    if master.empty:
        raise ValueError("Không có nhận xét trong phạm vi được chọn.")
    resources = segmenter_files(location(cfg, "vncorenlp_dir"))
    recorded = prep.get("segmenter_resource_sha256", {})
    if recorded and resources != recorded:
        raise ValueError("Tài nguyên VnCoreNLP khác lúc huấn luyện. Không trộn cách tách từ trong v01.")
    if args.check_only:
        status = environment()
        missing = [name for name in ["torch", "transformers", "py_vncorenlp", "java"] if not status.get(name)]
        if not resources:
            missing.append("vncorenlp_dir")
        print(json.dumps({"status": "model_and_data_valid", "reviews": len(master), "pairs": len(master) * 7,
                          "environment": status, "vncorenlp_resources_found": bool(resources),
                          "missing_for_new_text": missing,
                          "note": "Muốn xử lý văn bản chưa tách từ cần Java và py_vncorenlp."}, ensure_ascii=False, indent=2))
        if missing:
            raise RuntimeError("Chưa sẵn sàng dự đoán văn bản mới: " + ", ".join(missing))
        return
    status = environment()
    if not status["torch"] or not status["transformers"]:
        raise RuntimeError("Thiếu torch hoặc transformers trong Python đang dùng.")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = choose_device(args.device, allow_cpu=True)
    out = location(cfg, "prediction_dir")
    if args.scope != "all":
        out = out / args.scope
    if args.limit:
        out = out / f"smoke_{args.limit}"
    manifest_path = out / "prediction_manifest.json"
    chunk_size = int(cfg["inference_review_chunk_size"])
    identity_data = {"model_version": cfg["model_version"],
                     "model_completion_sha256": sha256_file(model_dir / "training_complete.json"),
                     "master_file_sha256": data_manifest["output_sha256"]["reviews_master.csv"],
                     "scope": args.scope, "limit": args.limit, "reviews": len(master),
                     "review_ids_sha256": object_hash(master.review_id.tolist()),
                     "review_chunk_size": chunk_size, "segmenter_resources": resources}
    identity = object_hash(identity_data)
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"Đã có lần dự đoán tại {out}. Thêm --resume; không tự ghi đè.")
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Dữ liệu/mô hình/phạm vi đã đổi. Không nối tiếp dự đoán khác phiên bản.")
        for i, part in enumerate(manifest["parts"]):
            if part["index"] != i or sha256_file(out / part["file"]) != part["sha256"]:
                raise ValueError("Phần dự đoán thiếu/đã đổi; không bỏ qua một cách im lặng.")
            validate_part(read_csv(out / part["file"]), master.iloc[i*chunk_size:(i+1)*chunk_size].review_id.tolist(),
                          cfg["model_version"], training_ids)
        if manifest.get("status") == "completed":
            if sha256_file(out / "aspect_predictions.csv") != manifest["output_sha256"]:
                raise ValueError("Tệp dự đoán tổng đã đổi.")
            print(f"Đã hoàn thành trước đó: {out / 'aspect_predictions.csv'}")
            return
    else:
        if args.resume:
            raise FileNotFoundError("Chưa có tiến độ dự đoán để tiếp tục. Bỏ --resume ở lần đầu.")
        out.mkdir(parents=True, exist_ok=True)
        if list((out / "parts").glob("*.csv")):
            raise FileExistsError("Có phần dự đoán nhưng mất manifest. Giữ lại để kiểm tra, không trộn vào lần mới.")
        manifest = {"created_at": now(), "status": "running", "identity": identity,
                    "identity_data": identity_data, "parts": [], "label_source": "model",
                    "confidence_calibrated": False, "manual_labels_modified": False,
                    "not_an_independent_evaluation": True}
        atomic_json(manifest_path, manifest)
    signature = object_hash({"preprocessing": prep, "resources": resources})
    cache = SegmentationCache(location(cfg, "system_data_dir") / "phobert_text_cache.sqlite3", signature)
    worker = None
    try:
        reused = seed_cache(cache, cfg, prep, master)
        print(f"Văn bản đã có thể tái sử dụng từ thực nghiệm: {reused}.", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
        model.eval()
        prompts = {a: tokenizer.encode(absa["prompts"][a], add_special_tokens=False) for a in ASPECTS}
        collator = PairCollator(tokenizer)
        pair_batch_size = args.batch_size or int(cfg["inference_batch_size"])
        window_batch_size = args.window_batch_size or int(cfg["inference_window_batch_size"])
        if min(pair_batch_size, window_batch_size) < 1:
            raise ValueError("Cỡ lô phải >=1.")
        amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        total_chunks = (len(master) + chunk_size - 1) // chunk_size
        for index in range(len(manifest["parts"]), total_chunks):
            rows = master.iloc[index * chunk_size:(index + 1) * chunk_size]
            examples = []
            for row in rows.itertuples(index=False):
                text = cache.get(row.review_text_nlp)
                if text is None:
                    if worker is None:
                        worker = Segmenter(location(cfg, "vncorenlp_dir"))
                    text = worker(row.review_text_nlp)
                    cache.put(row.review_text_nlp, text)
                tokens = tokenizer.encode(text, add_special_tokens=False)
                for aspect in ASPECTS:
                    examples.append({"review_id": row.review_id, "aspect": aspect,
                                     "has_manual_label": str(row.has_manual_label).lower(),
                                     "excluded_vlsp_overlap": str(row.excluded_vlsp_overlap).lower(),
                                     "windows": encode_pair(tokenizer, prompts[aspect], tokens,
                                                            int(absa["max_length"]), int(absa["overlap"]))})
            cache.commit()
            records = []
            with torch.inference_mode():
                for start in range(0, len(examples), pair_batch_size):
                    items = examples[start:start + pair_batch_size]
                    batch = collator(items)
                    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                        logits = aggregate_logits(model, batch, device, window_batch_size)
                    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                    for item, p in zip(items, probs):
                        order = np.argsort(-p, kind="stable")
                        rec = {"review_id": item["review_id"], "aspect": item["aspect"],
                               "predicted_label": LABELS[int(np.argmax(p))],
                               **{f"p_{label}": float(p[j]) for j, label in enumerate(LABELS)},
                               "confidence": float(p.max()), "margin": float(p[order[0]] - p[order[1]]),
                               "model_version": cfg["model_version"], "label_source": "model",
                               "has_manual_label": item["has_manual_label"],
                               "used_for_training": str(item["review_id"] in training_ids).lower(),
                               "excluded_vlsp_overlap": item["excluded_vlsp_overlap"],
                               "window_count": len(item["windows"])}
                        records.append(rec)
            frame = pd.DataFrame(records, columns=COLUMNS).astype(str)
            validate_part(frame, rows.review_id.tolist(), cfg["model_version"], training_ids)
            relative = f"parts/part_{index:05d}.csv"
            atomic_csv(out / relative, frame)
            manifest["parts"].append({"index": index, "file": relative,
                                      "reviews": len(rows), "rows": len(frame),
                                      "sha256": sha256_file(out / relative)})
            manifest["updated_at"] = now()
            atomic_json(manifest_path, manifest)
            print(f"Đã lưu phần {index+1}/{total_chunks}: {min((index+1)*chunk_size,len(master))}/{len(master)} nhận xét.", flush=True)
        count, digest = merge_parts(out, manifest, master, training_ids)
        manifest.update(status="completed", completed_at=now(), output_rows=count,
                        output_sha256=digest, output_file="aspect_predictions.csv")
        atomic_json(manifest_path, manifest)
        print(f"Hoàn thành {count} dòng dự đoán: {out / 'aspect_predictions.csv'}", flush=True)
    except KeyboardInterrupt:
        print("Đã dừng. Dùng lại cùng lệnh thêm --resume; chỉ phần chưa lưu sẽ chạy lại.", flush=True)
        raise SystemExit(130)
    finally:
        cache.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--scope", choices=["all", "unlabeled"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="Số nhận xét chạy thử; lưu riêng dưới smoke_N.")
    ap.add_argument("--batch-size", type=int, default=None, help="Cỡ lô cặp nhận xét–khía cạnh khi dự đoán.")
    ap.add_argument("--window-batch-size", type=int, default=None, help="Giới hạn số đoạn trong một lần forward.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check-env", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    if args.limit < 0:
        raise ValueError("--limit phải >=0.")
    if args.check_env:
        print(json.dumps(environment(), ensure_ascii=False, indent=2))
        return 0
    run(load_settings(args.config, args.project_root), args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"DỪNG: {exc}", file=sys.stderr)
        raise SystemExit(2)
