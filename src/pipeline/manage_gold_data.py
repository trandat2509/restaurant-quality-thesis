"""Ghép nhãn và bằng chứng tay vào bộ tích lũy; không huấn luyện hoặc kích hoạt system."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

from src.modeling.phobert_system_common import ASPECTS, load_settings, location, now, object_hash
from src.pipeline.prepare_system_data import (atomic_json, compact, csv_rows, flag, jread,
                                             load_system_inputs, require, sha)
from src.pipeline.update_system_data import read_batch, manual_changes, update_lock, write_csv

PAIR_FIELDS = ["review_id", "aspect", "label", "evidence_text", "evidence_spans_json",
               "label_verified", "evidence_verified", "uncertain", "annotation_version", "batch_id"]


def gold_paths(cfg):
    base = location(cfg, "system_data_dir").parent.parent / "annotations/foody"
    pointer = base / "current_gold.json"
    if not pointer.exists():
        return base, None
    saved = jread(pointer)
    from src.pipeline.update_system_data import safe_id
    folder = base / "versions" / safe_id(saved["batch_id"])
    require(sha(folder / "gold_manifest.json") == saved["manifest_sha256"], "Thông tin bộ gán tay đã bị đổi.")
    manifest = jread(folder / "gold_manifest.json")
    for name, digest in manifest["output_sha256"].items():
        p = (folder / name).resolve()
        require(p.is_relative_to(folder.resolve()) and p.is_file() and sha(p) == digest, "Tệp bộ gán tay thiếu/đã đổi: " + name)
    return base, (folder, manifest)


def prepare_gold(cfg, batch_path):
    batch = read_batch(batch_path)
    require(set(batch["sources"]) == {"manual_file"},
            "Đợt gán tay tích lũy chỉ dùng manual_file cho review đã có trong corpus hiện hành.")
    inputs = load_system_inputs(cfg["_root"], config=cfg["_config"])
    master = inputs["master"]
    base, previous = gold_paths(cfg)
    history = previous[1]["history"] if previous else []
    applied = next((h for h in history if h["batch_id"] == batch["batch_id"]), None)
    if applied:
        require(applied["identity"] == batch["identity"], "Mã đợt gán tay đã dùng với nội dung khác.")
        return {"already_applied": True, "batch": batch, "manifest": previous[1]}
    pairs = {}
    if previous:
        for row in csv_rows(previous[0] / "gold_pairs.csv"):
            pairs[(row["review_id"], row["aspect"])] = row
    else:
        # Khởi tạo từ tất cả nhãn tay gốc/đã nhập, bao gồm cả NONE.
        # Không lấy model_label hay bằng chứng đề xuất làm gold.
        for row in inputs["aspects"]:
            if row["label_source"] == "manual":
                key = (row["review_id"], row["aspect"])
                pairs[key] = {"review_id": key[0], "aspect": key[1], "label": row["manual_label"],
                              "evidence_text": row["evidence_text"], "evidence_spans_json": row["evidence_spans_json"],
                              "label_verified": row["label_verified"], "evidence_verified": row["evidence_verified"],
                              "uncertain": row["manual_uncertain"] or "0", "annotation_version": "legacy_manual",
                              "batch_id": "legacy_manual"}
    before_reviews = len({rid for rid, _ in pairs})
    changes = manual_changes(batch, master)
    for (rid, a), item in changes.items():
        old = pairs.get((rid, a))
        semantic_keys = ("label", "evidence_text", "evidence_spans_json", "label_verified", "evidence_verified")
        require(not old or all(old[k] == item[k] for k in semantic_keys)
                or batch["definition"].get("replace_manual", False),
                "Đang thay nhãn/bằng chứng đã lưu; cần replace_manual=true: " + rid + "/" + a)
        pairs[(rid, a)] = {"review_id": rid, "aspect": a, **{k: item[k] for k in semantic_keys},
                           "uncertain": item["manual_uncertain"],
                           "annotation_version": batch["definition"].get("annotation_version", "unspecified"),
                           "batch_id": batch["batch_id"]}
    # Mọi review và vị trí bằng chứng phải tiếp tục khớp corpus.
    from src.pipeline.update_system_data import check_spans
    for (rid, a), row in pairs.items():
        require(rid in master, "Gold có ID ngoài corpus đang dùng.")
        spans = json.loads(row["evidence_spans_json"])
        check_spans(master[rid]["review_text_nlp"], spans)
        if flag(row["evidence_verified"]):
            require([s["text"].strip() for s in spans] == [s.strip() for s in row["evidence_text"].split("||") if s.strip()],
                    "Bằng chứng đã xác nhận không khớp spans: " + rid + "/" + a)
    excluded = {rid for rid, r in master.items() if flag(r["excluded_vlsp_overlap"])}
    excluded_groups = {master[rid]["text_group_id"] for rid in excluded}
    wide, eligible, absa_eligible, evidence_pairs, pending = [], [], [], [], []
    fields = ["review_id", "text_group_id", "review_text_nlp", "label_status", "uncertain", *ASPECTS,
              *[a + "_evidence" for a in ASPECTS], *[a + "_evidence_spans_json" for a in ASPECTS]]
    for rid in sorted({rid for rid, _ in pairs}):
        current = {a: pairs[(rid, a)] for a in ASPECTS if (rid, a) in pairs}
        complete = len(current) == 7
        confirmed = complete and all(flag(r["label_verified"]) and flag(r["evidence_verified"]) and not flag(r["uncertain"]) for r in current.values())
        labels_confirmed = complete and all(flag(r["label_verified"]) and not flag(r["uncertain"]) for r in current.values())
        excluded_for_training = rid in excluded or master[rid]["text_group_id"] in excluded_groups
        if not excluded_for_training:
            for a, item in current.items():
                if flag(item["label_verified"]) and flag(item["evidence_verified"]) and not flag(item["uncertain"]):
                    evidence_pairs.append({**item, "text_group_id": master[rid]["text_group_id"],
                                           "review_text_nlp": master[rid]["review_text_nlp"]})
        if complete:
            row = {"review_id": rid, "text_group_id": master[rid]["text_group_id"],
                   "review_text_nlp": master[rid]["review_text_nlp"], "label_status": "Hoàn thành" if confirmed else "Cần xác nhận",
                   "uncertain": "0" if confirmed else "1", **{a: current[a]["label"] for a in ASPECTS},
                   **{a + "_evidence": current[a]["evidence_text"] for a in ASPECTS},
                   **{a + "_evidence_spans_json": current[a]["evidence_spans_json"] for a in ASPECTS}}
            wide.append(row)
            if confirmed and not excluded_for_training:
                eligible.append(row)
            if labels_confirmed and not excluded_for_training:
                absa_eligible.append({**row, "label_status": "Hoàn thành", "uncertain": "0"})
        if not confirmed or excluded_for_training:
            reason = ("excluded_overlap_group" if excluded_for_training else "incomplete_aspects" if not complete
                      else "unconfirmed_annotation" if any(not flag(r["label_verified"]) or flag(r["uncertain"]) for r in current.values())
                      else "unconfirmed_or_ambiguous_evidence")
            pending.append({"review_id": rid, "complete_aspects": len(current), "confirmed": str(confirmed).lower(),
                            "reason": reason})
    identity = object_hash({"pairs": [pairs[k] for k in sorted(pairs)],
                            "texts": {rid: master[rid]["review_text_nlp"] for rid, _ in pairs}})
    plan = {"gold_reviews_before": before_reviews, "gold_reviews_after": len({rid for rid, _ in pairs}),
            "complete_gold_reviews": len(wide), "training_candidate_reviews": len(eligible),
            "absa_candidate_reviews": len(absa_eligible), "evidence_candidate_pairs": len(evidence_pairs),
            "pending_or_excluded_reviews": len(pending), "manual_pairs_imported": len(changes),
            "pending_reason_counts": dict(Counter(r["reason"] for r in pending)),
            "model_retrained": False, "system_updated": False, "tier2_built": False,
            "corpus_reviews": len(master), "gold_dataset_sha256": identity}
    return {"already_applied": False, "batch": batch, "base": base, "previous": previous,
            "pairs": pairs, "wide": wide, "eligible": eligible, "pending": pending, "fields": fields,
            "absa_eligible": absa_eligible, "evidence_pairs": evidence_pairs,
            "history": history, "plan": plan, "master_sha256": sha(inputs["paths"]["master"])}


def apply_gold(cfg, batch_path):
    base = location(cfg, "system_data_dir")
    with update_lock(base):
        work = prepare_gold(cfg, batch_path)
        if work["already_applied"]:
            return {"status": "already_applied", "summary": work["manifest"]["summary"]}
        batch = work["batch"]
        versions = work["base"] / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        final = versions / batch["batch_id"]
        require(not final.exists(), "Thư mục đợt gán tay đã tồn tại; giữ để kiểm tra và dùng mã đợt khác.")
        temp = Path(tempfile.mkdtemp(prefix="building_", dir=versions))
        try:
            write_csv(temp / "gold_pairs.csv", PAIR_FIELDS, (work["pairs"][k] for k in sorted(work["pairs"])))
            write_csv(temp / "gold_all.csv", work["fields"], work["wide"])
            write_csv(temp / "training_candidates.csv", work["fields"], work["eligible"])
            write_csv(temp / "absa_training_candidates.csv", work["fields"], work["absa_eligible"])
            write_csv(temp / "evidence_training_pairs.csv", [*PAIR_FIELDS, "text_group_id", "review_text_nlp"], work["evidence_pairs"])
            write_csv(temp / "pending_reviews.csv", ["review_id", "complete_aspects", "confirmed", "reason"], work["pending"])
            shutil.copy2(batch["sources"]["manual_file"], temp / "manual_import.csv")
            require(sha(temp / "manual_import.csv") == batch["hashes"]["manual_file"], "Nhãn tay đổi trong khi ghép.")
            atomic_json(temp / "source_batch.json", batch["definition"])
            system_batch = {**batch["definition"], "manual_file": "manual_import.csv"}
            atomic_json(temp / "update_system_batch.json", system_batch)
            atomic_json(temp / "training_handoff.json", {
                "gold_dataset_sha256": work["plan"]["gold_dataset_sha256"],
                "candidate_file": "training_candidates.csv", "candidate_sha256": sha(temp / "training_candidates.csv"),
                "absa_candidate_file": "absa_training_candidates.csv", "absa_candidate_sha256": sha(temp / "absa_training_candidates.csv"),
                "evidence_pairs_file": "evidence_training_pairs.csv", "evidence_pairs_sha256": sha(temp / "evidence_training_pairs.csv"),
                "training_runs_automatically": False, "split_unit": "text_group_id",
                "evaluation_required_before_model_promotion": True,
                "independent_decisions": ["retrain_ABSA", "retrain_evidence"],
                "legacy_3350_experiments_unchanged": True,
                "note": "Tập ứng viên chưa chia train/validation/test. Chưa đo chất lượng và chưa được tự động huấn luyện."})
            outputs = [p.name for p in temp.iterdir() if p.is_file()]
            manifest = {"schema_version": 1, "created_at": now(), "summary": work["plan"],
                        "corpus_master_sha256": work["master_sha256"],
                        "history": [*work["history"], {"batch_id": batch["batch_id"], "identity": batch["identity"]}],
                        "output_sha256": {name: sha(temp / name) for name in outputs}}
            atomic_json(temp / "gold_manifest.json", manifest)
            require(read_batch(batch_path)["identity"] == batch["identity"], "Nguồn đổi khi đang ghép gold.")
            temp.rename(final)
            atomic_json(work["base"] / "current_gold.json", {"batch_id": batch["batch_id"],
                        "manifest_sha256": sha(final / "gold_manifest.json")})
            return {"status": "applied", "summary": work["plan"], "gold_directory": str(final),
                    "system_batch_file": str(final / "update_system_batch.json")}
        finally:
            if temp.exists():
                shutil.rmtree(temp)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--batch", required=True, type=Path)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args(argv)
    cfg = load_settings(args.config, args.project_root)
    work = apply_gold(cfg, args.batch) if args.apply else prepare_gold(cfg, args.batch)
    result = work if args.apply else ({"status": "already_applied"} if work["already_applied"] else work["plan"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit("DỪNG: " + str(exc))
