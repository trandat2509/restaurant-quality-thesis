"""Hàm dùng chung cho system tại foody_v01; giữ nguyên bộ mô hình và thực nghiệm.

prepared_data đọc nền huấn luyện đã chốt. load_system_inputs đọc nhãn/bằng chứng
vận hành mới trong cùng thư mục, không đưa nhãn máy trở lại tập huấn luyện.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ASPECTS = [f"A{i}" for i in range(1, 8)]
LABELS = ["NONE", "POS", "NEU", "NEG", "MIX"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def fingerprint(df):
    columns = ["review_id", "text_group_id", "review_text_nlp"] + ASPECTS
    return object_hash(df[columns].astype(str).sort_values("review_id").to_dict("records"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_csv(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_csv(name, index=False, encoding="utf-8-sig")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def resolve(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def load_settings(config=None, project_root=None):
    root = Path(project_root).resolve() if project_root else ROOT
    path = resolve(root, config or "configs/system/foody_v01.json")
    raw = path.read_bytes()
    cfg = json.loads(raw.decode("utf-8-sig"))
    cfg["_root"] = root
    cfg["_config"] = path
    cfg["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def location(cfg, key):
    return resolve(cfg["_root"], cfg[key])


def require_columns(df, columns, label):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label}: thiếu cột {sorted(missing)}.")
    for col in columns:
        if df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label}: cột {col} có ô trống; không tự loại dòng.")


def validate_reviews(df, label, with_labels=False):
    columns = ["review_id", "text_group_id", "review_text_nlp"] + (ASPECTS if with_labels else [])
    require_columns(df, columns, label)
    if df.empty or df.review_id.duplicated().any():
        raise ValueError(f"{label}: dữ liệu trống hoặc review_id trùng.")
    if with_labels:
        for aspect in ASPECTS:
            if not df[aspect].isin(LABELS).all():
                raise ValueError(f"{label}: nhãn không hợp lệ ở {aspect}.")


def environment():
    result = {m: importlib.util.find_spec(m) is not None
              for m in ["torch", "transformers", "py_vncorenlp", "pandas", "numpy", "sentencepiece"]}
    result["java"] = shutil.which("java")
    if result["torch"]:
        import torch
        result.update(torch_version=torch.__version__, cuda=torch.cuda.is_available(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    return result


def reference_config(cfg):
    ref = read_json(location(cfg, "reference_run_config"))
    if ref.get("labels") != LABELS or ref.get("pooling") != "mean_logits_over_all_review_windows":
        raise ValueError("Cấu hình tham chiếu không đúng kiến trúc 5 lớp/gộp logits.")
    if not ref.get("encoder_init") or ref["encoder_init"].get("classifier_transferred") is not False:
        raise ValueError("Tham chiếu phải dùng encoder VLSP và bộ phân loại mới.")
    actual = location(cfg, "actual_cv_run_config")
    if actual.exists():
        other = read_json(actual)
        keys = ["dataset_sha256", "preprocessing_sha256", "model_name", "epochs", "batch_size",
                "accumulation_steps", "learning_rate", "weight_decay", "max_length", "overlap",
                "weighting", "seed", "pooling", "labels"]
        mismatch = [k for k in keys if ref.get(k) != other.get(k)]
        if ref["encoder_init"]["checkpoint_sha256"] != other.get("encoder_init", {}).get("checkpoint_sha256"):
            mismatch.append("encoder_init.checkpoint_sha256")
        if mismatch:
            raise ValueError("Báo cáo trên máy khác cấu hình đã chốt: " + ", ".join(mismatch))
    return ref


def prepared_data(cfg):
    base = location(cfg, "system_data_dir")
    manifest = read_json(base / "data_manifest.json")
    # Đầu ra vận hành có vòng cập nhật riêng ở system_manifest.json.
    # Việc sửa bằng chứng không được làm thay đổi nguồn của mô hình đã học.
    for name in ["train_reviews.csv", "reviews_master.csv"]:
        if name not in manifest["output_sha256"]:
            raise ValueError(f"Báo cáo nền thiếu mã kiểm tra: {name}")
        expected = manifest["output_sha256"][name]
        p = base / name
        if not p.exists() or sha256_file(p) != expected:
            raise ValueError(f"Dữ liệu chuẩn bị đã đổi/thiếu: {p}. Chạy prepare_system_data.")
    train = read_csv(base / "train_reviews.csv")
    master = read_csv(base / "reviews_master.csv")
    validate_reviews(train, "train_reviews", True)
    validate_reviews(master, "reviews_master")
    if fingerprint(train) != manifest["training_dataset_sha256"]:
        raise ValueError("Dấu vân tay dữ liệu huấn luyện không khớp.")
    return train, master, manifest


def load_system_inputs(project_root=ROOT, view="operational", config=None):
    """Đầu đọc thống nhất cho giao diện/tầng cảnh báo, không chạy lại mô hình."""
    from src.pipeline.prepare_system_data import load_system_inputs as load_current
    return load_current(project_root=project_root, view=view, config=config)


def preprocessing_reference(cfg, train, ref):
    path = location(cfg, "preprocessing_file")
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {path}. Cần giữ lại tệp tách từ đã dùng khi chạy 5 phần.")
    if sha256_file(path) != ref["preprocessing_sha256"]:
        raise ValueError("phobert_preprocessing.json không khớp lần thực nghiệm đã chốt.")
    prep = read_json(path)
    if prep.get("dataset_sha256") != fingerprint(train) or not prep.get("word_segmentation_applied"):
        raise ValueError("Tham chiếu tách từ không khớp dữ liệu/không có tách từ.")
    if set(prep.get("prompts", {})) != set(ASPECTS):
        raise ValueError("Thiếu nội dung khía cạnh đã tách từ.")
    seg_path = location(cfg, "segmented_file")
    if not seg_path.exists() or sha256_file(seg_path) != prep["segmented_file_sha256"]:
        raise ValueError("Tệp absa_phobert_segmented.csv thiếu hoặc đã đổi.")
    segmented = read_csv(seg_path)
    require_columns(segmented, ["review_id", "phobert_text"], "Dữ liệu tách từ")
    if segmented.review_id.duplicated().any() or set(segmented.review_id) != set(train.review_id):
        raise ValueError("ID dữ liệu tách từ không khớp dữ liệu huấn luyện.")
    if "review_text_nlp" in segmented:
        texts = train.set_index("review_id").review_text_nlp
        if not segmented.review_text_nlp.eq(segmented.review_id.map(texts)).all():
            raise ValueError("Văn bản tách từ không thuộc văn bản huấn luyện hiện tại.")
    return prep, segmented.set_index("review_id").phobert_text.to_dict()


def encoder_reference(cfg, ref):
    path = location(cfg, "encoder_dir")
    source = path / "source_training_manifest.json"
    if not source.exists():
        raise FileNotFoundError(f"Thiếu checkpoint VLSP đầy đủ: {source}")
    info = read_json(source)
    if not info.get("training_completed") or info.get("transfer_component") != "encoder_only":
        raise ValueError("Checkpoint không phải encoder VLSP đã hoàn thành.")
    if info.get("test_used") is not False or info.get("labels") != LABELS[:4]:
        raise ValueError("Nguồn VLSP không khớp chính sách/bộ nhãn 4 lớp.")
    if info.get("model_name") != ref["model_name"]:
        raise ValueError("Tên mô hình nguồn không khớp.")
    expected = ref["encoder_init"]["checkpoint_sha256"]
    for name, digest in expected.items():
        if not (path / name).exists() or sha256_file(path / name) != digest:
            raise ValueError(f"Checkpoint VLSP sai/thiếu tệp: {name}")
    if sha256_file(source) != ref["encoder_init"]["manifest_sha256"]:
        raise ValueError("Manifest checkpoint VLSP khác lần thực nghiệm.")
    return path, expected


def token_windows(tokens, capacity, overlap):
    if capacity < 1 or overlap < 0 or overlap >= capacity:
        raise ValueError("Độ dài đoạn/chồng lấn không hợp lệ.")
    if not tokens:
        return [[]]
    result, start = [], 0
    while True:
        result.append(tokens[start:start + capacity])
        if start + capacity >= len(tokens):
            return result
        start += capacity - overlap


def encode_pair(tokenizer, prompt_ids, review_ids, max_length, overlap):
    capacity = max_length - len(prompt_ids) - tokenizer.num_special_tokens_to_add(pair=True)
    result = []
    for window in token_windows(review_ids, capacity, overlap):
        ids = tokenizer.build_inputs_with_special_tokens(prompt_ids, window)
        if len(ids) > max_length:
            raise ValueError("Đoạn vượt giới hạn.")
        result.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return result


def label_weights(counts, weighting):
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (7, 5) or (counts < 0).any() or (counts.sum(axis=1) == 0).any():
        raise ValueError("Phân bố nhãn không hợp lệ.")
    weights = np.ones_like(counts)
    if weighting == "sqrt_balanced":
        for i in range(7):
            present = counts[i] > 0
            weights[i, present] = np.clip(np.sqrt(counts[i].sum() /
                                         (present.sum() * counts[i, present])), .25, 4.)
    elif weighting != "none":
        raise ValueError("Kiểu cân bằng không hợp lệ.")
    return weights


class PairCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        import torch
        windows, owner = [], []
        for i, item in enumerate(items):
            windows.extend(item["windows"])
            owner.extend([i] * len(item["windows"]))
        encoded = self.tokenizer.pad(windows, padding=True, return_tensors="pt")
        return {"encoded": encoded, "owner": torch.tensor(owner, dtype=torch.long), "items": items}


def aggregate_logits(model, batch, device, window_batch_size=None):
    import torch
    encoded = {k: v.to(device) for k, v in batch["encoded"].items()
               if k in {"input_ids", "attention_mask"}}
    count = encoded["input_ids"].shape[0]
    size = window_batch_size or count
    logits = torch.cat([model(**{k: v[start:start + size] for k, v in encoded.items()}).logits.float()
                        for start in range(0, count, size)], dim=0)
    owner = batch["owner"].to(device)
    total = torch.zeros((len(batch["items"]), len(LABELS)), dtype=logits.dtype, device=device)
    total = total.index_add(0, owner, logits)
    return total / torch.bincount(owner, minlength=len(batch["items"])).unsqueeze(1)


class Segmenter:
    def __init__(self, directory):
        if importlib.util.find_spec("py_vncorenlp") is None:
            raise RuntimeError("Thiếu py_vncorenlp; cài vào đúng môi trường Python đang dùng.")
        if not shutil.which("java"):
            raise RuntimeError("Chưa tìm thấy Java trong PATH.")
        directory = Path(directory).resolve()
        if not directory.is_dir() or not list(directory.glob("*.jar")):
            raise FileNotFoundError(f"Không thấy VnCoreNLP tại {directory}. Sửa vncorenlp_dir trong cấu hình.")
        import py_vncorenlp
        previous = Path.cwd()
        try:
            self.worker = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(directory))
        finally:
            os.chdir(previous)

    def __call__(self, text):
        previous = Path.cwd()
        try:
            result = " ".join(self.worker.word_segment(unicodedata.normalize("NFC", str(text)))).strip()
        finally:
            os.chdir(previous)
        if not result:
            raise ValueError("VnCoreNLP trả văn bản trống.")
        return result


def segmenter_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {str(p.relative_to(directory)).replace("\\", "/"): sha256_file(p)
            for p in sorted(directory.rglob("*")) if p.is_file()
            and (p.suffix.lower() in {".jar", ".rdr", ".dict"} or "wordsegmenter" in str(p).lower())}


def choose_device(requested, allow_cpu=False):
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA chưa sẵn sàng trong Python đang dùng.")
    device = "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        raise RuntimeError("Không có GPU. Chỉ thêm --allow-cpu khi chủ động chấp nhận chạy CPU.")
    return device
