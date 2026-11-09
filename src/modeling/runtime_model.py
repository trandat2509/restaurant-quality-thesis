"""Đọc mô hình vận hành độc lập với tập thí nghiệm 3.350 đã đóng băng."""
from pathlib import Path

from src.modeling.phobert_system_common import ASPECTS, LABELS, location
from src.pipeline.prepare_system_data import jread, csv_rows, require, sha


def model_bundle(cfg, profile):
    model_dir = location(dict(cfg, model_dir=profile["model_dir"]), "model_dir")
    done = jread(model_dir / "training_complete.json")
    require(done.get("training_completed") is True, "Mô hình chưa huấn luyện hoàn tất.")
    hashes = done.get("files_sha256", {})
    required = {"config.json", "model.safetensors", "absa_config.json", "preprocessing.json", "training_review_ids.csv"}
    require(required <= set(hashes), "Mô hình thiếu danh mục kiểm tra trọng số/cấu hình/tập đã học.")
    for name, digest in hashes.items():
        path = (model_dir / name).resolve()
        require(path.is_relative_to(model_dir) and path.is_file() and sha(path) == digest,
                "Tệp mô hình thiếu hoặc thay đổi: " + name)
    absa, prep = jread(model_dir / "absa_config.json"), jread(model_dir / "preprocessing.json")
    require(absa.get("model_version") == profile["model_version"], "Phiên mô hình khác hồ sơ khai báo.")
    require(absa.get("aspects") == ASPECTS and absa.get("labels") == LABELS and
            absa.get("pooling") == "mean_logits_over_all_review_windows", "Mô hình khác bộ A1–A7/nhãn/cách gộp đã chốt.")
    require(set(absa.get("prompts", {})) == set(ASPECTS), "Thiếu lời nhắc khía cạnh của mô hình.")
    ids = [r["review_id"] for r in csv_rows(model_dir / "training_review_ids.csv")]
    require(ids and len(ids) == len(set(ids)), "Danh sách nhận xét đã học trống hoặc trùng.")
    identity = {"model_version": profile["model_version"], "model_dir": profile["model_dir"],
                "vncorenlp_dir": profile.get("vncorenlp_dir", cfg["vncorenlp_dir"]),
                "completion_sha256": sha(model_dir / "training_complete.json"),
                "training_review_ids": sorted(ids),
                "training_dataset_sha256": absa.get("training_dataset_sha256", "")}
    if profile.get("completion_sha256"):
        require(profile["completion_sha256"] == identity["completion_sha256"],
                "Mô hình vận hành đã bị thay đổi; phải đăng ký phiên mới và dự đoán lại.")
    return model_dir, absa, prep, set(ids), identity


def prediction_config(cfg, profile=None):
    runtime = dict(cfg)
    if profile:
        runtime.update({k: profile[k] for k in ("model_version", "model_dir", "vncorenlp_dir") if k in profile})
        runtime["_runtime_model"] = profile
    return runtime
