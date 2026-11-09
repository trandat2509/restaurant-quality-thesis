"""Chuẩn bị VLSP riêng biệt; không chuyển nhãn VLSP thành A1–A7."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .common import ASPECTS, LABELS, DEFAULT_DATA, load_experiment, sha256_file, write_json

SOURCE_PROMPTS = {
    "AMBIENCE#GENERAL": "Không gian và bầu không khí của nhà hàng",
    "DRINKS#PRICES": "Giá đồ uống",
    "DRINKS#QUALITY": "Chất lượng đồ uống",
    "DRINKS#STYLE&OPTIONS": "Cách trình bày và sự lựa chọn đồ uống",
    "FOOD#PRICES": "Giá món ăn",
    "FOOD#QUALITY": "Chất lượng món ăn",
    "FOOD#STYLE&OPTIONS": "Cách trình bày khẩu phần và sự lựa chọn món ăn",
    "LOCATION#GENERAL": "Vị trí của nhà hàng",
    "RESTAURANT#GENERAL": "Đánh giá chung về nhà hàng",
    "RESTAURANT#MISCELLANEOUS": "Các nội dung khác về nhà hàng",
    "RESTAURANT#PRICES": "Giá cả chung của nhà hàng",
    "SERVICE#GENERAL": "Dịch vụ và phục vụ của nhà hàng",
}
SOURCE_ASPECTS = list(SOURCE_PROMPTS)
SOURCE_LABELS = ["NONE", "POS", "NEU", "NEG"]
CSV_LABEL_MAP = {0: "NONE", 1: "POS", 2: "NEG", 3: "NEU"}
VERSION = "vlsp_transfer_prepare_v02.0"


def match_key(text):
    text = unicodedata.normalize("NFC", str(text)).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def find_source(directory, split):
    candidates = sorted(Path(directory).rglob(f"*-VLSP2018-SA-Restaurant-{split}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Không thấy CSV Restaurant-{split} trong {directory}")
    if len({sha256_file(p) for p in candidates}) != 1:
        raise ValueError(f"Có nhiều bản {split} khác nội dung: {[str(p) for p in candidates]}")
    return candidates[0]


def read_source(path, split):
    d = pd.read_csv(path, keep_default_na=False)
    if set(d.columns) != {"Review", *SOURCE_ASPECTS}:
        raise ValueError(f"Sai cột của CSV gốc: {path}")
    if d.empty or d.Review.astype(str).str.strip().eq("").any():
        raise ValueError(f"Review trống trong {path}")
    result = pd.DataFrame({"source_id": [f"vlsp_{split}_{i+1:06d}" for i in range(len(d))],
                           "source_split": split, "csv_row": np.arange(len(d)) + 2,
                           "review_text_raw": d.Review.astype(str)})
    for aspect in SOURCE_ASPECTS:
        values = pd.to_numeric(d[aspect], errors="raise")
        if not values.isin(CSV_LABEL_MAP).all():
            raise ValueError(f"Sai nhãn 0/1/2/3 ở {path}: {aspect}")
        result[aspect] = values.map(CSV_LABEL_MAP)
    return result


def near_matches(left, right, threshold=0.90, min_chars=40):
    """Heuristic trùng gần, không học đặc trưng cho mô hình phân loại."""
    li = [i for i, s in enumerate(left) if len(s) >= min_chars]
    ri = [i for i, s in enumerate(right) if len(s) >= min_chars]
    if not li or not ri:
        return []
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32)
    matrix = vectorizer.fit_transform([left[i] for i in li] + [right[i] for i in ri])
    a, b = matrix[:len(li)], matrix[len(li):]
    result = []
    for start in range(0, len(li), 128):
        similarities = (a[start:start+128] @ b.T).tocoo()
        for row, col, score in zip(similarities.row, similarities.col, similarities.data):
            if score >= threshold:
                i, j = li[start + int(row)], ri[int(col)]
                if min(len(left[i]), len(right[j])) / max(len(left[i]), len(right[j])) >= 0.80:
                    result.append((i, j, float(score)))
    return result


def audit_and_clean(frames, foody, threshold=0.90):
    d = pd.concat(frames, ignore_index=True)
    keys = d.review_text_raw.map(match_key)
    reasons = {i: set() for i in d.index}
    links = []
    cross_groups = conflict_groups = 0
    for key, indices in keys.groupby(keys).groups.items():
        ids = list(indices)
        if not key:
            for i in ids: reasons[i].add("empty_after_matching_normalization")
        if len(ids) < 2:
            continue
        g = d.loc[ids]
        cross = g.source_split.nunique() > 1
        conflict = len(g[SOURCE_ASPECTS].drop_duplicates()) > 1
        cross_groups += int(cross)
        conflict_groups += int(conflict)
        for i in ids:
            # Test gốc được giữ nguyên; chỉ loại nội dung chồng lấn khỏi train/dev.
            if d.at[i, "source_split"] == "test":
                continue
            if cross:
                reasons[i].add("source_cross_split_duplicate")
            if conflict:
                reasons[i].add("source_duplicate_conflicting_labels")
            if not cross and not conflict and i != ids[0]:
                reasons[i].add("source_same_split_duplicate")
    target_keys = foody.review_text_nlp.map(match_key).tolist()
    target_lookup = {}
    for j, key in enumerate(target_keys):
        target_lookup.setdefault(key, []).append(str(foody.iloc[j].review_id))
    for i, key in enumerate(keys):
        if key and key in target_lookup:
            if d.at[i, "source_split"] != "test": reasons[i].add("foody_normalized_text_overlap")
            for rid in target_lookup[key]:
                links.append({"source_id": d.at[i, "source_id"], "other_id": rid,
                              "kind": "foody_normalized_text_overlap", "similarity": 1.0})
    for i, j, score in near_matches(keys.tolist(), target_keys, threshold):
        if keys.iloc[i] == target_keys[j]: continue
        if d.at[i, "source_split"] != "test": reasons[i].add("foody_near_text_overlap")
        links.append({"source_id": d.at[i, "source_id"], "other_id": str(foody.iloc[j].review_id),
                      "kind": "foody_near_text_overlap", "similarity": score})
    train_indices = d.index[d.source_split == "train"].tolist()
    for other_split in ["dev", "test"]:
        other_indices = d.index[d.source_split == other_split].tolist()
        for i, j, score in near_matches(keys.iloc[train_indices].tolist(), keys.iloc[other_indices].tolist(), threshold):
            a, b = train_indices[i], other_indices[j]
            if keys.iloc[a] == keys.iloc[b]: continue
            reasons[a].add(f"source_near_overlap_{other_split}")
            if other_split == "dev": reasons[b].add("source_near_overlap_train")
            links.append({"source_id": d.at[a, "source_id"], "other_id": d.at[b, "source_id"],
                          "kind": f"source_near_overlap_{other_split}", "similarity": score})
    excluded = d.loc[[i for i in d.index if reasons[i]]].copy()
    excluded["exclusion_reason"] = [";".join(sorted(reasons[i])) for i in excluded.index]
    clean = d.loc[[i for i in d.index if not reasons[i]]].copy()
    for split in ["train", "dev"]:
        if not (clean.source_split == split).any(): raise ValueError(f"Tập {split} rỗng sau kiểm tra.")
    return clean, excluded, pd.DataFrame(links, columns=["source_id", "other_id", "kind", "similarity"]), {
        "duplicate_groups_across_splits_loose_normalization": cross_groups,
        "duplicate_groups_with_conflicting_labels": conflict_groups,
        "near_duplicate_threshold": threshold,
        "note": "Chuẩn hóa để so trùng bỏ dấu câu/emoji, giữ dấu tiếng Việt. Gần trùng là heuristic, không bảo đảm tìm mọi diễn đạt tương đương. Test gốc không dùng để học/chọn epoch."}


def segment_source(frames, tools_dir):
    if not shutil.which("java"):
        raise SystemExit("Chưa tìm thấy Java trong PATH. Dùng môi trường đã chạy PhoBERT v01.")
    import py_vncorenlp
    previous = Path.cwd()
    try:
        segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(tools_dir))
        def segment(text):
            value = " ".join(segmenter.word_segment(unicodedata.normalize("NFC", str(text)))).strip()
            if not value: raise ValueError("Tách từ trả văn bản trống.")
            return value
        prompts = {a: segment("Khía cạnh: " + s) for a, s in SOURCE_PROMPTS.items()}
        for split, d in frames.items():
            if split == "test": continue
            output = []
            for i, text in enumerate(d.review_text_raw):
                output.append(segment(text))
                if (i + 1) % 200 == 0: print(f"Tách từ VLSP {split}: {i+1}/{len(d)}", flush=True)
            d["phobert_text"] = output
    finally:
        os.chdir(previous)
    return prompts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default="data/external/vlsp2018/raw")
    ap.add_argument("--foody-data-dir", default=DEFAULT_DATA)
    ap.add_argument("--output-dir", default="data/external/vlsp2018/processed_v02")
    ap.add_argument("--report-dir", default="reports/modeling/vlsp_data_audit_v02")
    ap.add_argument("--vncorenlp-dir", default="tools/vncorenlp")
    ap.add_argument("--segment", action="store_true", help="Tách từ train/dev để chuẩn bị huấn luyện; không có cờ này chỉ kiểm tra và làm sạch.")
    ap.add_argument("--near-threshold", type=float, default=0.90)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not 0.8 <= args.near_threshold <= 1: raise ValueError("near-threshold phải từ 0.8 đến 1.")
    source_dir, data_dir, out, report, vn = [Path(p).resolve() for p in [args.source_dir, args.foody_data_dir, args.output_dir, args.report_dir, args.vncorenlp_dir]]
    confirmed, meta = load_experiment(data_dir)
    all_path = data_dir / meta.get("all_reviews_file", "absa_350_all.csv")
    if not all_path.exists(): raise FileNotFoundError("Thiếu tệp toàn bộ review khai báo trong split_manifest.json để loại trùng VLSP.")
    foody = pd.read_csv(all_path, dtype=str, keep_default_na=False)
    if not {"review_id", "review_text_nlp"}.issubset(foody.columns): raise ValueError("Thiếu cột Foody.")
    if not set(confirmed.review_id).issubset(set(foody.review_id)): raise ValueError("Tệp toàn bộ review không bao phủ tập đang học.")
    sources = {s: find_source(source_dir, s) for s in ["train", "dev", "test"]}
    manifest = {"version": VERSION, "source_sha256": {s: sha256_file(p) for s, p in sources.items()},
                "foody_dataset_sha256": meta["dataset_sha256"], "foody_split_sha256": meta["split_sha256"],
                "foody_all_sha256": sha256_file(all_path), "segment": args.segment,
                "near_threshold": args.near_threshold, "source_labels": SOURCE_LABELS,
                "csv_label_map": CSV_LABEL_MAP, "source_aspects": SOURCE_ASPECTS,
                "original_text_modified": False, "foody_labels_modified": False,
                "test_used_for_training_or_selection": False}
    manifest_path = out / "preparation_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        same = all(old.get(k) == json.loads(json.dumps(v)) for k, v in manifest.items())
        intact = all((out / n).exists() and sha256_file(out / n) == h for n, h in old.get("output_sha256", {}).items())
        if same and intact:
            print(f"Dữ liệu đã chuẩn bị, mã kiểm tra khớp: {out}"); return
        raise FileExistsError("Đầu ra đã có nhưng khác cấu hình/nội dung. Chọn thư mục mới hoặc dùng --overwrite có chủ ý.")
    frames = [read_source(p, split) for split, p in sources.items()]
    clean, excluded, overlaps, audit = audit_and_clean(frames, foody, args.near_threshold)
    splits = {s: clean[clean.source_split == s].reset_index(drop=True) for s in sources}
    prompts = segment_source(splits, vn) if args.segment else SOURCE_PROMPTS.copy()
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    for split, d in splits.items(): d.to_csv(out / f"{split}.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(report / "excluded_source_rows.csv", index=False, encoding="utf-8-sig")
    overlaps.to_csv(report / "overlap_pairs.csv", index=False, encoding="utf-8-sig")
    distribution = []
    for split, d in splits.items():
        for a in SOURCE_ASPECTS:
            for label in SOURCE_LABELS: distribution.append({"split": split, "aspect": a, "label": label, "count": int(d[a].eq(label).sum())})
    pd.DataFrame(distribution).to_csv(report / "source_label_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{ "aspect": a, **{label: int(confirmed[a].eq(label).sum()) for label in LABELS}}
                  for a in ASPECTS]).to_csv(report / "foody_label_coverage.csv", index=False, encoding="utf-8-sig")
    audit.update({"original_rows": {s: len(d) for s, d in zip(sources, frames)},
                  "retained_rows": {s: len(d) for s, d in splits.items()}, "excluded_rows": len(excluded),
                  "foody_reviews_checked": len(foody), "word_segmentation_completed": args.segment})
    write_json(report / "audit_summary.json", audit)
    manifest["prompts"] = prompts
    manifest["rows"] = {s: len(d) for s, d in splits.items()}
    manifest["output_sha256"] = {f"{s}.csv": sha256_file(out / f"{s}.csv") for s in splits}
    write_json(manifest_path, manifest)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not args.segment: print("CHƯA TÁCH TỪ: muốn huấn luyện, chạy lại với --segment --overwrite trong môi trường PhoBERT.")


if __name__ == "__main__":
    main()
