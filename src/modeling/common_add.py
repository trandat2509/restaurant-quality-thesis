from __future__ import annotations

import hashlib
import json
import platform
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ASPECTS = [f"A{i}" for i in range(1, 8)]
LABELS = ["NONE", "POS", "NEU", "NEG", "MIX"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
DEFAULT_DATA = "data/modeling/foody_add"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip().lower()


def fingerprint(df):
    cols = ["review_id", "text_group_id", "review_text_nlp"] + ASPECTS
    records = df[cols].astype(str).sort_values("review_id").to_dict("records")
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_dataset(df):
    required = ["review_id", "text_group_id", "review_text_nlp"] + ASPECTS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")
    if df.empty:
        raise ValueError("Tập dữ liệu trống.")
    for col in required:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Cột {col} có ô trống.")
    if df.review_id.duplicated().any():
        raise ValueError("review_id trùng; không tự xóa dòng có nhãn.")
    for aspect in ASPECTS:
        if not df[aspect].isin(LABELS).all():
            raise ValueError(f"Nhãn không hợp lệ ở {aspect}.")


def split_groups(df):
    """Gộp cùng text_group_id hoặc cùng văn bản chuẩn hóa để ngăn rò rỉ."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    seen_groups, seen_texts = {}, {}
    for i, row in enumerate(df.itertuples(index=False)):
        for seen, key in [(seen_groups, str(row.text_group_id)),
                          (seen_texts, normalized_text(row.review_text_nlp))]:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    components = {}
    for i in range(len(df)):
        components.setdefault(find(i), []).append(str(df.iloc[i].review_id))
    names = {root: "G_" + hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:16]
             for root, ids in components.items()}
    return [names[find(i)] for i in range(len(df))]


def token_windows(token_ids, capacity, overlap):
    if capacity < 1 or overlap < 0 or overlap >= capacity:
        raise ValueError("Dung lượng/độ chồng lấn đoạn không hợp lệ.")
    if not token_ids:
        return [[]]
    result = []
    start = 0
    while True:
        result.append(token_ids[start:start + capacity])
        if start + capacity >= len(token_ids):
            return result
        start += capacity - overlap


def masked_softmax(logits, supported):
    values = np.asarray(logits, dtype=float).copy()
    values[:, ~np.asarray(supported, dtype=bool)] = -np.inf
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def prediction_records(df, aspect, probabilities, model_name, fold):
    order = np.argsort(-probabilities, axis=1)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        rec = {"review_id": row.review_id, "text_group_id": row.text_group_id,
               "fold": int(fold), "aspect": aspect, "gold": getattr(row, aspect),
               "pred": LABELS[int(order[i, 0])], "confidence": float(probabilities[i, order[i, 0]]),
               "margin": float(probabilities[i, order[i, 0]] - probabilities[i, order[i, 1]]),
               "model": model_name}
        rec.update({f"p_{label}": float(probabilities[i, j]) for j, label in enumerate(LABELS)})
        rows.append(rec)
    return rows


def report_predictions(predictions, out_dir, expected_df=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = predictions.copy()
    if pred.duplicated(["review_id", "aspect"]).any():
        raise ValueError("Một cặp review–khía cạnh có nhiều dự đoán kiểm tra chéo.")
    if expected_df is not None:
        expected = {(rid, a) for rid in expected_df.review_id for a in ASPECTS}
        if set(zip(pred.review_id, pred.aspect)) != expected:
            raise ValueError("Thiếu/thừa dự đoán; chưa hoàn thành đủ 5 lượt.")
    aspect_rows, class_rows, fold_rows, class_fold_rows = [], [], [], []
    for aspect in ASPECTS:
        a = pred[pred.aspect == aspect]
        gold, guessed = a.gold.to_numpy(), a.pred.to_numpy()
        supported = [label for label in LABELS if (a.gold == label).any()]
        non_none = [label for label in supported if label != "NONE"]
        p, r, f, n = precision_recall_fscore_support(gold, guessed, labels=LABELS, zero_division=0)
        for j, label in enumerate(LABELS):
            class_rows.append({"aspect": aspect, "label": label, "precision": p[j], "recall": r[j],
                               "f1": f[j], "support": int(n[j]), "insufficient_support": int(n[j] < 5)})
        true_presence = gold != "NONE"
        pred_presence = guessed != "NONE"
        pp, rr, ff, _ = precision_recall_fscore_support(true_presence, pred_presence, average="binary", zero_division=0)
        aspect_rows.append({"aspect": aspect, "accuracy": accuracy_score(gold, guessed),
                            "macro_f1_5_labels": f1_score(gold, guessed, labels=LABELS, average="macro", zero_division=0),
                            "weighted_f1": f1_score(gold, guessed, labels=LABELS, average="weighted", zero_division=0),
                            "macro_f1_supported": f1_score(gold, guessed, labels=supported, average="macro", zero_division=0),
                            "non_none_macro_f1": f1_score(gold, guessed, labels=non_none, average="macro", zero_division=0) if non_none else 0.,
                            "presence_precision": pp, "presence_recall": rr, "presence_f1": ff,
                            "review_count": len(a)})
        cm = pd.DataFrame(confusion_matrix(gold, guessed, labels=LABELS), index=LABELS, columns=LABELS)
        cm.index.name = "gold_label"
        cm.to_csv(out_dir / f"confusion_matrix_{aspect}_add.csv", encoding="utf-8-sig")
        for fold in sorted(a.fold.unique()):
            b = a[a.fold == fold]
            fold_rows.append({"fold": int(fold), "aspect": aspect, "test_reviews": len(b),
                              "macro_f1_5_labels": f1_score(b.gold, b.pred, labels=LABELS, average="macro", zero_division=0),
                              "weighted_f1": f1_score(b.gold, b.pred, labels=LABELS, average="weighted", zero_division=0),
                              "presence_f1": f1_score(b.gold != "NONE", b.pred != "NONE", zero_division=0)})
            fp, fr, ff1, fn = precision_recall_fscore_support(b.gold, b.pred, labels=LABELS, zero_division=0)
            for j, label in enumerate(LABELS):
                class_fold_rows.append({"fold": int(fold), "aspect": aspect, "label": label,
                                       "precision": fp[j], "recall": fr[j], "f1": ff1[j], "support": int(fn[j])})
    per_aspect, per_fold = pd.DataFrame(aspect_rows), pd.DataFrame(fold_rows)
    for metric in ("macro_f1_5_labels", "weighted_f1", "presence_f1"):
        stats = per_fold.groupby("aspect")[metric].agg(["mean", "std"]).fillna(0.)
        per_aspect[f"{metric}_fold_mean"] = per_aspect.aspect.map(stats["mean"])
        per_aspect[f"{metric}_fold_std"] = per_aspect.aspect.map(stats["std"])
    per_label, per_label_fold = pd.DataFrame(class_rows), pd.DataFrame(class_fold_rows)
    for metric in ("precision", "recall", "f1"):
        stats = per_label_fold.groupby(["aspect", "label"])[metric].agg(["mean", "std"]).fillna(0.)
        lookup = pd.MultiIndex.from_frame(per_label[["aspect", "label"]])
        per_label[f"{metric}_fold_mean"] = stats["mean"].reindex(lookup).to_numpy()
        per_label[f"{metric}_fold_std"] = stats["std"].reindex(lookup).to_numpy()
    per_aspect.to_csv(out_dir / "metrics_by_aspect_add.csv", index=False, encoding="utf-8-sig")
    per_label.to_csv(out_dir / "metrics_by_label_add.csv", index=False, encoding="utf-8-sig")
    per_label_fold.to_csv(out_dir / "metrics_by_label_fold_add.csv", index=False, encoding="utf-8-sig")
    per_fold.to_csv(out_dir / "metrics_by_fold_add.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(out_dir / "out_of_fold_predictions_add.csv", index=False, encoding="utf-8-sig")
    errors = pred[pred.gold != pred.pred]
    if expected_df is not None:
        errors = errors.merge(expected_df[["review_id", "review_text_nlp"]], on="review_id", validate="many_to_one")
    errors.to_csv(out_dir / "prediction_errors_add.csv", index=False, encoding="utf-8-sig")
    exact = pred.assign(correct=pred.gold == pred.pred).groupby("review_id").correct.all().mean()
    fold_macro = per_fold.groupby("fold").macro_f1_5_labels.mean()
    summary = {"model": str(pred.model.iloc[0]), "review_count": int(pred.review_id.nunique()),
               "prediction_count": len(pred), "macro_f1_7_aspects_5_labels": float(per_aspect.macro_f1_5_labels.mean()),
               "weighted_f1_7_aspects": float(per_aspect.weighted_f1.mean()),
               "macro_f1_7_aspects_supported_labels": float(per_aspect.macro_f1_supported.mean()),
               "micro_f1_all_aspect_labels": float(f1_score(pred.gold, pred.pred, average="micro")),
               "exact_review_accuracy": float(exact), "fold_macro_f1_mean": float(fold_macro.mean()),
               "fold_macro_f1_std": float(fold_macro.std(ddof=1)) if len(fold_macro) > 1 else 0.,
               "error_count": len(errors), "python": platform.python_version(),
               "notes": ["Đây là kiểm tra chéo phát triển, chưa phải tập kiểm tra độc lập cuối cùng.",
                         "Macro 5 nhãn cố định tính F1=0 cho lớp không có ví dụ; đọc kèm support và macro supported.",
                         "Micro F1 gộp nhiều NONE; không dùng làm chỉ số duy nhất.",
                         "confidence là xác suất mô hình, chưa được hiệu chuẩn thành độ đúng thực tế."]}
    write_json(out_dir / "metrics_summary_add.json", summary)
    return summary
