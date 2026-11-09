"""Dự đoán đúng phần nhận xét mới bằng mô hình ABSA đã có; không huấn luyện."""
from __future__ import annotations

from pathlib import Path
import json


def predict_reviews(cfg, records, output_path, device="auto"):
    # Các thư viện mô hình chỉ được nhập khi người dùng chọn chạy dự đoán.
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.modeling.predict_phobert import COLUMNS, SegmentationCache, model_metadata, validate_part
    from src.modeling.phobert_system_common import (
        ASPECTS, LABELS, PairCollator, Segmenter, aggregate_logits, atomic_csv, atomic_json,
        choose_device, encode_pair, location, object_hash, prepared_data, read_csv, read_json,
        segmenter_files, sha256_file,
    )
    from src.pipeline.prepare_system_data import text_hash, require

    if cfg.get("_runtime_model"):
        from src.modeling.runtime_model import model_bundle
        model_dir, absa, prep, training_ids, _ = model_bundle(cfg, cfg["_runtime_model"])
    else:
        train, _, base = prepared_data(cfg)
        model_dir, _, absa, prep, training_ids = model_metadata(cfg)
        require(training_ids == set(train.review_id) and absa["training_dataset_sha256"] == base["training_dataset_sha256"],
                "Mô hình không thuộc tập huấn luyện nền đang giữ.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resource_hashes = segmenter_files(location(cfg, "vncorenlp_dir"))
    require(not prep.get("segmenter_resource_sha256") or prep["segmenter_resource_sha256"] == resource_hashes,
            "Tài nguyên tách từ khác lúc huấn luyện.")
    signature = object_hash({"model": sha256_file(model_dir / "training_complete.json"),
                             "preprocessing": prep, "resources": resource_hashes,
                             "records": [(r["review_id"], text_hash(r["review_text_nlp"])) for r in records]})
    proof_path = output_path.with_suffix(".proof.json")
    if proof_path.exists():
        proof = read_json(proof_path)
        require(proof["signature"] == signature, "Đợt dự đoán đã có dùng nguồn hoặc mô hình khác; dùng mã đợt mới.")
        if proof.get("complete"):
            require(output_path.is_file() and sha256_file(output_path) == proof["output_sha256"], "Tệp dự đoán đã lưu bị đổi.")
            return output_path
    else:
        proof = {"signature": signature, "complete": False, "parts": []}
        atomic_json(proof_path, proof)
    device = choose_device(device, allow_cpu=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()
    prompts = {a: tokenizer.encode(absa["prompts"][a], add_special_tokens=False) for a in ASPECTS}
    collator = PairCollator(tokenizer)
    cache = SegmentationCache(output_path.parent / "tokens.sqlite3", object_hash({"prep": prep, "resources": resource_hashes}))
    worker = None
    chunk_size = int(cfg["inference_review_chunk_size"])
    pair_batch = int(cfg["inference_batch_size"])
    window_batch = int(cfg["inference_window_batch_size"])
    require(min(chunk_size, pair_batch, window_batch) > 0, "Kích thước lô phải dương.")
    parts = output_path.parent / "prediction_parts"
    parts.mkdir(exist_ok=True)
    known = {p["index"]: p for p in proof["parts"]}
    frames = []
    try:
        for index, offset in enumerate(range(0, len(records), chunk_size)):
            chunk = records[offset:offset + chunk_size]
            target = parts / f"part_{index:05d}.csv"
            if index in known:
                require(target.is_file() and sha256_file(target) == known[index]["sha256"], "Phần dự đoán đã lưu bị đổi.")
                frame = read_csv(target)
                validate_part(frame[COLUMNS], [r["review_id"] for r in chunk], cfg["model_version"], training_ids)
                expected = {r["review_id"]: text_hash(r["review_text_nlp"]) for r in chunk}
                require(all(r.review_text_sha256 == expected[r.review_id] for r in frame.itertuples()), "Văn bản của phần dự đoán đã đổi.")
                frames.append(frame)
                continue
            examples = []
            texts = {r["review_id"]: r["review_text_nlp"] for r in chunk}
            for row in chunk:
                segmented = cache.get(row["review_text_nlp"])
                if segmented is None:
                    if worker is None:
                        worker = Segmenter(location(cfg, "vncorenlp_dir"))
                    segmented = worker(row["review_text_nlp"])
                    cache.put(row["review_text_nlp"], segmented)
                tokens = tokenizer.encode(segmented, add_special_tokens=False)
                for aspect in ASPECTS:
                    examples.append({"review_id": row["review_id"], "aspect": aspect,
                                     "windows": encode_pair(tokenizer, prompts[aspect], tokens, int(absa["max_length"]), int(absa["overlap"])),
                                     "has_manual_label": str(row.get("has_manual_label", "false")).lower(),
                                     "excluded_vlsp_overlap": str(row.get("excluded_vlsp_overlap", "false")).lower()})
            cache.commit()
            output = []
            with torch.inference_mode():
                for start in range(0, len(examples), pair_batch):
                    items = examples[start:start + pair_batch]
                    batch = collator(items)
                    probabilities = torch.softmax(aggregate_logits(model, batch, device, window_batch).float(), dim=1).cpu().numpy()
                    for item, values in zip(items, probabilities):
                        order = np.argsort(-values, kind="stable")
                        output.append({"review_id": item["review_id"], "aspect": item["aspect"],
                                       "predicted_label": LABELS[int(values.argmax())],
                                       **{f"p_{label}": float(values[i]) for i, label in enumerate(LABELS)},
                                       "confidence": float(values.max()), "margin": float(values[order[0]] - values[order[1]]),
                                       "model_version": cfg["model_version"], "label_source": "model",
                                       "has_manual_label": item["has_manual_label"], "used_for_training": str(item["review_id"] in training_ids).lower(),
                                       "excluded_vlsp_overlap": item["excluded_vlsp_overlap"], "window_count": len(item["windows"]),
                                       "review_text_sha256": text_hash(texts[item["review_id"]])})
            frame = pd.DataFrame(output, columns=[*COLUMNS, "review_text_sha256"]).astype(str)
            validate_part(frame[COLUMNS], [r["review_id"] for r in chunk], cfg["model_version"], training_ids)
            atomic_csv(target, frame)
            proof["parts"].append({"index": index, "sha256": sha256_file(target)})
            atomic_json(proof_path, proof)
            frames.append(frame)
            print(f"Đã dự đoán {min(offset + chunk_size, len(records))}/{len(records)} nhận xét cần bổ sung.", flush=True)
        atomic_csv(output_path, pd.concat(frames, ignore_index=True))
        proof.update(complete=True, output_sha256=sha256_file(output_path), model_retrained=False)
        atomic_json(proof_path, proof)
        return output_path
    finally:
        cache.close()
