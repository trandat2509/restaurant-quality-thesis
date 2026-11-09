from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

LABELS = ["O", "B", "I"]
LABEL2ID = {v: i for i, v in enumerate(LABELS)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError("Cần tên cột khi ghi bảng rỗng")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def occurrences(text, fragment):
    result, start = [], 0
    while True:
        start = text.find(fragment, start)
        if start < 0:
            return result
        result.append([start, start + len(fragment)])
        start += 1


def merge_spans(spans):
    result = []
    for start, end in sorted(set(tuple(s) for s in spans)):
        if end <= start:
            raise ValueError("Đoạn rỗng hoặc có vị trí sai")
        if result and start < result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def canonical_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def assign_groups(rows):
    # Hợp nhất cả nhóm đã có và văn bản chuẩn hóa trùng nhau.
    parent = {r["review_id"]: r["review_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x, y = find(x), find(y)
        if x != y:
            parent[max(x, y)] = min(x, y)

    known_groups, known_texts = {}, {}
    for row in rows:
        rid = row["review_id"]
        keys = ((known_groups, row.get("text_group_id", "")),
                (known_texts, canonical_text(row["review_text_nlp"])))
        for table, key in keys:
            if key:
                if key in table:
                    union(rid, table[key])
                else:
                    table[key] = rid
    return {rid: find(rid) for rid in parent}


def group_split(groups, fractions, seed):
    unique = sorted(set(groups.values()))
    random.Random(seed).shuffle(unique)
    n_test = max(1, round(len(unique) * fractions["test"]))
    n_val = max(1, round(len(unique) * fractions["validation"]))
    if n_test + n_val >= len(unique):
        raise ValueError("Không đủ nhóm cho ba phần dữ liệu")
    group_part = {g: "test" for g in unique[:n_test]}
    group_part.update({g: "validation" for g in unique[n_test:n_test+n_val]})
    group_part.update({g: "train" for g in unique[n_test+n_val:]})
    return {rid: group_part[g] for rid, g in groups.items()}


def encode_bio(offsets, spans):
    labels = [0] * len(offsets)
    for i, (start, end) in enumerate(offsets):
        if end <= start:
            labels[i] = -100
    for s, e in spans:
        found = [i for i, (start, end) in enumerate(offsets)
                 if start < e and end > s and end > start]
        if not found:
            raise ValueError(f"Bằng chứng [{s}, {e}) không có đơn vị văn bản tương ứng")
        for j, i in enumerate(found):
            labels[i] = 1 if j == 0 else 2
    return labels


def decode_bio(offsets, labels, text):
    spans, active = [], None
    for (start, end), label in zip(offsets, labels):
        if end <= start:
            continue
        if label == 0 or label == -100:
            if active is not None:
                spans.append(active)
                active = None
        elif label == 1 or active is None:
            if active is not None:
                spans.append(active)
            active = [start, end]
        else:
            active[1] = max(active[1], end)
    if active is not None:
        spans.append(active)
    result = []
    for start, end in merge_spans(spans):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end-1].isspace():
            end -= 1
        if end > start:
            result.append([start, end])
    return result


def char_set(spans, text):
    return {i for s, e in spans for i in range(s, e) if not text[i].isspace()}


def evaluate_spans(records):
    # Tính theo đoạn chính xác và ký tự, gộp sau khi ghép các cửa sổ.
    totals = Counter()
    aspects = {}
    for row in records:
        gold = set(tuple(s) for s in row["gold_spans"])
        pred = set(tuple(s) for s in row["predicted_spans"])
        gc, pc = char_set(gold, row["text"]), char_set(pred, row["text"])
        c = Counter({"pairs": 1, "positive_pairs": int(bool(gold)),
                     "span_tp": len(gold & pred), "span_pred": len(pred), "span_gold": len(gold),
                     "char_tp": len(gc & pc), "char_pred": len(pc), "char_gold": len(gc),
                     "exact_pair": int(gold == pred),
                     "positive_exact_pair": int(bool(gold) and gold == pred),
                     "empty_pair": int(not gold), "empty_false_positive": int(not gold and bool(pred))})
        totals.update(c)
        aspects.setdefault(row["aspect"], Counter()).update(c)

    def finish(c):
        v = dict(c)
        for key in ["span", "char"]:
            p = c[key+"_tp"] / c[key+"_pred"] if c[key+"_pred"] else 0.0
            r = c[key+"_tp"] / c[key+"_gold"] if c[key+"_gold"] else 0.0
            v.update({key+"_precision": p, key+"_recall": r,
                      key+"_f1": 2*p*r/(p+r) if p+r else 0.0})
        v["exact_pair_rate"] = c["exact_pair"] / max(c["pairs"], 1)
        v["positive_exact_pair_rate"] = c["positive_exact_pair"] / max(c["positive_pairs"], 1)
        v["empty_false_positive_rate"] = c["empty_false_positive"] / max(c["empty_pair"], 1)
        return v
    return {"overall": finish(totals), "by_aspect": {a: finish(c) for a, c in aspects.items()}}


def make_windows(ids, prefix_ids, labels, max_length=256, overlap=32, cls=0, sep=2):
    budget = max_length - len(prefix_ids) - 4
    if budget <= overlap or overlap < 0:
        raise ValueError("Độ dài cửa sổ hoặc độ chồng lấn không hợp lệ")
    result, start = [], 0
    while start < len(ids):
        end = min(len(ids), start + budget)
        prefix = [cls] + list(prefix_ids) + [sep, sep]
        result.append({"input_ids": prefix + ids[start:end] + [sep],
                       "labels": [-100]*len(prefix) + labels[start:end] + [-100],
                       "start": start, "end": end, "prefix_length": len(prefix)})
        if end == len(ids):
            break
        start = end - overlap
    return result
