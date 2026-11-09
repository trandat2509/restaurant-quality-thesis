"""Tầng 2: xu hướng tiêu cực theo nhà hàng, tháng lịch và khía cạnh.

Đọc bản system đang dùng, không huấn luyện hoặc dự đoán. Điều kiện độ phủ
đọc riêng từ configs/alerts/tang2/trend_alerts.json; không sửa cấu hình system nền.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.phobert_system_common import ASPECTS, LABELS, load_settings, location
from src.pipeline.prepare_system_data import load_system_inputs, system_paths
from src.pipeline.prepare_system_data import LAYOUT_VERSION, alert_output_dir

VERSION = "foody_trend_alerts_v05_manual_priority"
CONFIG = "configs/alerts/tang2/trend_alerts.json"
ASPECT_NAMES = {"A1": "Thái độ phục vụ", "A2": "Thời gian phục vụ", "A3": "Chất lượng món ăn",
                "A4": "Không gian và cơ sở vật chất", "A5": "Vệ sinh", "A6": "Giá cả",
                "A7": "Tiếp nhận và xử lý phản hồi"}
STATUS_NAMES = {
    "eligible": "Đủ dữ liệu để đánh giá", "open_period": "Tháng chưa kết thúc",
    "missing_current": "Tháng đang xét không có nhận xét",
    "insufficient_current": "Tháng đang xét chưa đủ dữ liệu",
    "missing_baseline": "Thiếu tháng trong đường cơ sở",
    "insufficient_baseline": "Tháng trong đường cơ sở chưa đủ dữ liệu",
    "awaiting_predictions": "Chưa đủ dự đoán máy thống nhất",
}
LEVEL_NAMES = {"red": "Cảnh báo đỏ", "yellow": "Cảnh báo vàng", "none": "Chưa đạt ngưỡng cảnh báo",
               "not_evaluated": "Chưa đủ điều kiện đánh giá"}
COVERAGE_FIELDS = ["restaurant_id", "restaurant_name", "month", "review_count", "text_group_count",
                   "baseline_months", "closed_period", "baseline_periods_json", "coverage_eligible", "status", "status_title",
                   "reason", "profile", "temporary", "model_version"]
METRIC_FIELDS = ["restaurant_id", "restaurant_name", "month", "aspect", "aspect_name", "review_count",
                 "text_group_count", "baseline_months", *["count_" + x for x in LABELS], "complaint_rate", "negative_content_rate",
                 "negative_among_mentions_rate", "wilson_lower", "wilson_upper", "baseline_rate",
                 "increase_pp", "baseline_periods_json", "eligible", "status", "reason", "level", "level_title",
                 "negative_review_ids_json", "profile", "temporary", "model_version"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def month_number(month):
    y, m = map(int, month.split("-"))
    require(1 <= m <= 12 and len(month) == 7, "Tháng không hợp lệ: " + month)
    return y * 12 + m - 1


def month_name(number):
    return f"{number // 12:04d}-{number % 12 + 1:02d}"


def previous_months(month, window=3):
    n = month_number(month)
    return [month_name(n - i) for i in range(window, 0, -1)]


def as_of_time(value=None, zone="Asia/Ho_Chi_Minh"):
    tz = ZoneInfo(zone)
    if value is None:
        return datetime.now(tz)
    if isinstance(value, datetime):
        require(value.tzinfo is not None, "Thời điểm đánh giá cần có múi giờ.")
        return value.astimezone(tz)
    if len(str(value)) == 10:
        return datetime.combine(date.fromisoformat(str(value)), time.max, tzinfo=tz)
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Thời điểm đánh giá cần có múi giờ.")
    return result.astimezone(tz)


def policy_for(cfg, profile=None):
    path = cfg["_root"] / CONFIG
    settings = read_json(path)
    require(settings.get("schema_version") == 2 and settings.get("enabled") is True,
            "Tầng 2 chưa bật trong " + CONFIG)
    selected = profile or settings["active_profile"]
    require(selected in settings["profiles"], "Không có bộ điều kiện: " + str(selected))
    entry = settings["profiles"][selected]
    policy = deepcopy(settings["policy"])
    require(policy["primary_period"] == "month" and policy["baseline_windows"] == [2, 3],
            "Tầng 2 dùng hai cửa sổ 2 và 3 tháng lịch liền trước.")
    require(policy["baseline_implementation"] == "arithmetic_mean_of_previous_calendar_period_rates"
            and policy["counting_unit"] == "review_id" and policy["primary_metric"] == "complaint_rate",
            "Công thức tầng 2 khác phương án đã chốt.")
    require(policy["manual_labels_replace_predictions"] is True
            and policy.get("label_selection") == "confirmed_manual_else_model"
            and policy["model_confidence_filter"] is None,
            "Tầng 2 ưu tiên nhãn tay xác nhận, còn lại dùng nhãn máy; không lọc theo điểm mô hình.")
    keys = {"min_reviews_current", "min_reviews_each_baseline_period",
            "min_distinct_text_groups_current", "min_distinct_text_groups_each_baseline_period"}
    require(set(entry["data_requirements"]) == keys, "Bộ điều kiện độ phủ thiếu/thừa khóa.")
    require(all(type(v) is int and v > 0 for v in entry["data_requirements"].values()),
            "Điều kiện độ phủ phải là số nguyên dương.")
    policy["data_requirements"].update(entry["data_requirements"])
    require(policy["data_requirements"]["require_closed_current_period"] is True
            and policy["data_requirements"]["zero_review_period_is_missing"] is True,
            "Không tính tháng chưa kết thúc hoặc coi tháng thiếu là tỷ lệ 0.")
    require(policy["uncertainty"]["affects_level"] is False, "Khoảng Wilson chỉ để trình bày.")
    for name in ("yellow", "red"):
        t = policy["thresholds"][name]
        require(0 <= t["min_current_rate"] <= 1 and 0 <= t["min_increase_pp"] <= 100
                and type(t["min_problem_reviews"]) is int and t["min_problem_reviews"] >= 1,
                "Ngưỡng cảnh báo không hợp lệ.")
    require(all(policy["thresholds"]["red"][k] >= policy["thresholds"]["yellow"][k]
                for k in policy["thresholds"]["yellow"]), "Ngưỡng đỏ phải không thấp hơn vàng.")
    policy.update(profile=selected, profile_title=entry["title"], temporary=bool(entry["temporary"]))
    return policy


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1-p) / n + z*z / (4*n*n)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def alert_level(rate, increase_pp, negative_count, eligible, thresholds):
    if not eligible:
        return "not_evaluated"
    for level in ("red", "yellow"):
        t = thresholds[level]
        if (rate + 1e-12 >= t["min_current_rate"] and increase_pp + 1e-10 >= t["min_increase_pp"]
                and negative_count >= t["min_problem_reviews"]):
            return level
    return "none"


def coverage_text(policy):
    """Mô tả đúng điều kiện của tháng xét và từng tháng cơ sở trong cấu hình."""
    req = policy["data_requirements"]
    current = (req["min_reviews_current"], req["min_distinct_text_groups_current"])
    baseline = (req["min_reviews_each_baseline_period"], req["min_distinct_text_groups_each_baseline_period"])
    if current == baseline:
        return (f"Tháng đang xét và từng tháng cơ sở đều cần ít nhất {current[0]} nhận xét "
                f"và {current[1]} nhóm văn bản khác nhau.")
    return (f"Tháng đang xét cần ít nhất {current[0]} nhận xét và {current[1]} nhóm văn bản khác nhau. "
            f"Từng tháng cơ sở cần ít nhất {baseline[0]} nhận xét và {baseline[1]} nhóm văn bản khác nhau.")


def threshold_text(policy):
    """Dùng chung quy tắc màu cho báo cáo, HTML và phần xem trong notebook."""
    parts = []
    for level, title in (("yellow", "Vàng"), ("red", "Đỏ")):
        t = policy["thresholds"][level]
        parts.append(f"{title}: tỷ lệ ≥{100 * t['min_current_rate']:g}%, "
                     f"tăng ≥{t['min_increase_pp']:g} điểm phần trăm, "
                     f"ít nhất {t['min_problem_reviews']} NEG.")
    return " ".join(parts)


def aggregate(inputs, policy, as_of=None):
    """Giữ nhãn máy riêng; thống kê bằng nhãn tay xác nhận hoặc nhãn máy."""
    cutoff = as_of_time(as_of, policy["timezone"])
    current_month = cutoff.strftime("%Y-%m")
    all_master = inputs["master"]
    master = {rid: r for rid, r in all_master.items()
              if datetime.fromisoformat(r["review_date_iso"].replace("Z", "+00:00")) <= cutoff}
    period_reviews, groups, restaurants = defaultdict(list), defaultdict(set), {}
    for rid, r in master.items():
        require(r.get("text_group_id"), "Thiếu mã nhóm văn bản: " + rid)
        local_month = datetime.fromisoformat(r["review_date_iso"].replace("Z", "+00:00")).astimezone(ZoneInfo(policy["timezone"])).strftime("%Y-%m")
        require(r["review_month"] == local_month, "Tháng không khớp múi giờ Việt Nam: " + rid)
        key = r["restaurant_id"], local_month
        period_reviews[key].append(rid)
        groups[key].add(r["text_group_id"])
        restaurants[r["restaurant_id"]] = r["restaurant_name"]
    counts, negatives, labels = defaultdict(Counter), defaultdict(list), {}
    evidence = {}
    seen, versions, missing = set(), set(), set()
    selected_sources = Counter()
    for r in inputs["aspects"]:
        rid, a = r["review_id"], r["aspect"]
        require(rid in all_master and a in ASPECTS, "Cặp dự đoán ngoài đầu vào system.")
        require((rid, a) not in seen, "Trùng cặp dự đoán máy.")
        seen.add((rid, a))
        machine_label = r.get("model_label", "")
        if not machine_label:
            missing.add(rid)
            continue
        confirmed = (r.get("label_source") == "manual"
                     and str(r.get("label_verified", "")).lower() in {"true", "1"})
        label = (r.get("manual_label") or r.get("label")) if confirmed else machine_label
        require(label in LABELS and machine_label in LABELS and r.get("model_version"), "Nhãn hoặc phiên mô hình không hợp lệ.")
        versions.add(r["model_version"])
        if (r.get("evidence_text") and r.get("label") == label
                and ((confirmed and r.get("evidence_source") == "manual")
                     or (not confirmed and r.get("evidence_source") == "model" and r.get("evidence_model_version")))):
            evidence[(rid, a)] = {k: r.get(k, "") for k in
                                 ("evidence_text", "evidence_model_version", "evidence_status", "evidence_source")}

        if rid not in master:
            continue
        selected_sources['manual' if confirmed else 'model'] += 1
        meta = master[rid]
        key = meta["restaurant_id"], meta["review_month"], a
        counts[key][label] += 1
        labels[(rid, a)] = label
        if label == "NEG":
            negatives[key].append(rid)
    require(len(versions) <= 1, "System đang trộn nhiều phiên mô hình; chưa tính tầng 2.")
    for rid in all_master:
        if any((rid, a) not in seen for a in ASPECTS):
            missing.add(rid)
    expected_version = inputs["manifest"].get("current_model_version")
    require(not expected_version or not versions or versions == {expected_version},
            "Phiên dự đoán không khớp mô hình đang vận hành.")
    model_version = next(iter(versions), expected_version or "")
    ready = not missing and inputs["manifest"].get("summary", {}).get("pending_model_reviews", 0) == 0
    req = policy["data_requirements"]
    coverage, metrics, alerts = [], [], []
    first_month = {}
    for restaurant, month in period_reviews:
        first_month[restaurant] = min(first_month.get(restaurant, month_number(month)), month_number(month))
    for restaurant in sorted(restaurants):
        for n in range(first_month[restaurant], month_number(current_month) + 1):
            month = month_name(n)
            key = restaurant, month
            total, group_count = len(period_reviews[key]), len(groups[key])
            for window in policy["baseline_windows"]:
                baseline = [{"month": m, "review_count": len(period_reviews[restaurant, m]),
                             "text_group_count": len(groups[restaurant, m])} for m in previous_months(month, window)]
                current_ok = total >= req["min_reviews_current"] and group_count >= req["min_distinct_text_groups_current"]
                baseline_ok = all(b["review_count"] >= req["min_reviews_each_baseline_period"] and
                                  b["text_group_count"] >= req["min_distinct_text_groups_each_baseline_period"] for b in baseline)
                coverage_ok = month < current_month and current_ok and baseline_ok
                status = ("awaiting_predictions" if not ready else "open_period" if month >= current_month
                          else "missing_current" if total == 0 else "insufficient_current" if not current_ok
                          else "missing_baseline" if any(b["review_count"] == 0 for b in baseline)
                          else "insufficient_baseline" if not baseline_ok else "eligible")
                deficiencies = []
                if month >= current_month:
                    deficiencies.append("Tháng " + month + " chưa kết thúc")
                if not current_ok:
                    deficiencies.append(f"{month}: {total} nhận xét/{group_count} nhóm; cần {req['min_reviews_current']}/{req['min_distinct_text_groups_current']}")
                for b in baseline:
                    if b["review_count"] < req["min_reviews_each_baseline_period"] or b["text_group_count"] < req["min_distinct_text_groups_each_baseline_period"]:
                        deficiencies.append(f"{b['month']}: {b['review_count']} nhận xét/{b['text_group_count']} nhóm; cần {req['min_reviews_each_baseline_period']}/{req['min_distinct_text_groups_each_baseline_period']}")
                if not ready:
                    deficiencies.insert(0, "Còn thiếu dự đoán máy; không tính tỷ lệ hoặc phát cảnh báo trên tập thiếu")
                reason = "; ".join(deficiencies) if deficiencies else f"Đạt điều kiện tháng hiện tại và từng tháng trong {window} tháng cơ sở"
                common = dict(restaurant_id=restaurant, restaurant_name=restaurants[restaurant], month=month,
                              review_count=total, text_group_count=group_count, baseline_months=window, status=status, reason=reason,
                              profile=policy["profile"], temporary=policy["temporary"], model_version=model_version)
                coverage.append({**common, "closed_period": month < current_month, "baseline_periods_json": compact(baseline),
                                 "coverage_eligible": coverage_ok, "status_title": STATUS_NAMES[status]})
                if total == 0:
                    continue  # Tháng trống có trong bảng độ phủ, không sinh tỷ lệ 0 giả.
                for a in ASPECTS:
                    cs = counts[restaurant, month, a]
                    if ready:
                        require(sum(cs.values()) == total, "Mẫu số khía cạnh không khớp tổng nhận xét.")
                    rate = cs["NEG"] / total if ready else None
                    content_rate = (cs["NEG"] + cs["MIX"]) / total if ready else None
                    mentioned = total - cs["NONE"]
                    among = (cs["NEG"] + cs["MIX"]) / mentioned if ready and mentioned else None
                    lo, hi = wilson(cs["NEG"], total, policy["uncertainty"]["wilson_z"]) if ready else (None, None)
                    bs = [{**b, "negative_count": counts[restaurant, b["month"], a]["NEG"] if ready and b["review_count"] else None,
                           "complaint_rate": counts[restaurant, b["month"], a]["NEG"] / b["review_count"] if ready and b["review_count"] else None}
                          for b in baseline]
                    base_rate = sum(b["complaint_rate"] for b in bs) / window if ready and coverage_ok and all(b["complaint_rate"] is not None for b in bs) else None
                    change = (rate - base_rate) * 100 if rate is not None and base_rate is not None else None
                    eligible = status == "eligible"
                    level = alert_level(rate, change, cs["NEG"], eligible, policy["thresholds"])
                    row = {**common, "aspect": a, "aspect_name": ASPECT_NAMES[a],
                           **{"count_" + x: cs[x] if ready else None for x in LABELS},
                           "complaint_rate": rate, "negative_content_rate": content_rate, "negative_among_mentions_rate": among,
                           "wilson_lower": lo, "wilson_upper": hi, "baseline_rate": base_rate, "increase_pp": change,
                           "baseline_periods_json": compact(bs), "eligible": eligible,
                           "level": level, "level_title": LEVEL_NAMES[level],
                           "negative_review_ids_json": compact(sorted(negatives[restaurant, month, a])) if ready else "[]"}
                    metrics.append(row)
                    if level in {"red", "yellow"}:
                        alerts.append(row)
    alerts.sort(key=lambda r: (r["month"], r["level"] == "red", r["increase_pp"], r["restaurant_id"], r["aspect"]), reverse=True)
    observed = {(r["restaurant_id"], r["month"]) for r in coverage if r["review_count"]}
    eligible_keys = {(r["restaurant_id"], r["month"]) for r in coverage if r["status"] == "eligible"}
    coverage_keys = {(r["restaurant_id"], r["month"]) for r in coverage if r["coverage_eligible"]}
    eligible = [r for r in coverage if r["status"] == "eligible"]
    summary = {"reviews": len(master), "excluded_future_reviews": len(all_master)-len(master), "restaurants": len(restaurants),
               "observed_restaurant_months": len(observed), "calendar_restaurant_months": len({(r["restaurant_id"], r["month"]) for r in coverage}),
               "coverage_eligible_restaurant_months": len(coverage_keys),
               "eligible_restaurant_months": len(eligible_keys), "eligible_restaurants": len({r["restaurant_id"] for r in eligible}),
               "eligible_aspect_comparisons": sum(r["eligible"] for r in metrics), "aspect_metric_rows": len(metrics),
               "alert_rows": len(alerts), "alert_restaurant_months": len({(r["restaurant_id"], r["month"]) for r in alerts}),
               "alert_restaurants": len({r["restaurant_id"] for r in alerts}), "levels": dict(Counter(r["level"] for r in alerts)),
               "coverage_statuses": dict(Counter(r["status"] for r in coverage)), "model_coverage_ready": ready,
               "pending_model_reviews": len(missing), "model_version": model_version,
               "profile": policy["profile"], "temporary": policy["temporary"], "as_of": cutoff.isoformat(),
               "latest_review_month": max((r["review_month"] for r in master.values()), default=None),
               "latest_closed_month": month_name(month_number(current_month)-1),
               "latest_closed_month_alert_rows": sum(r["month"] == month_name(month_number(current_month)-1) for r in alerts),
               "new_independent_model_evaluation": False,
               "baseline_windows": policy["baseline_windows"],
               "eligible_window_comparisons": len(eligible),
               "alert_unique_events": len({(r["restaurant_id"], r["month"], r["aspect"]) for r in alerts}),
               "label_selection": "confirmed_manual_else_model", "selected_label_sources": dict(selected_sources),
               "by_window": {str(w): {
                   "eligible_restaurant_months": sum(r["baseline_months"] == w for r in eligible),
                   "eligible_restaurants": len({r["restaurant_id"] for r in eligible if r["baseline_months"] == w}),
                   "alert_rows": sum(r["baseline_months"] == w for r in alerts),
                   "alert_restaurants": len({r["restaurant_id"] for r in alerts if r["baseline_months"] == w}),
                   "levels": dict(Counter(r["level"] for r in alerts if r["baseline_months"] == w))
               } for w in policy["baseline_windows"]}}
    return {"coverage": coverage, "metrics": metrics, "alerts": alerts, "summary": summary,
            "policy": policy, "master": master, "labels": labels, "evidence": evidence}


def report_text(result):
    s, p = result["summary"], result["policy"]
    lines = ["# Kết quả cảnh báo tầng 2", "", f"Ngày đánh giá: {s['as_of']}. Mô hình: {s['model_version']}.",
             "", p["profile_title"], "", "| Cửa sổ cơ sở | Nhà hàng–tháng đủ điều kiện | Nhà hàng có cảnh báo | Lượt cảnh báo |",
             "|---|---:|---:|---:|"]
    for window, v in s["by_window"].items():
        lines.append(f"| {window} tháng | {v['eligible_restaurant_months']} | {v['alert_restaurants']} | {v['alert_rows']} |")
    lines += ["", f"Tổng {s['reviews']} nhận xét, {s['restaurants']} nhà hàng. Có {s['alert_restaurants']} nhà hàng có cảnh báo.",
              f"Có {s['alert_unique_events']} tổ hợp nhà hàng–tháng–khía cạnh có cảnh báo, {s['alert_rows']} lượt khi tính cả hai cửa sổ.",
              "", "Mỗi nhà hàng hiển thị một lần; các tháng, khía cạnh và cửa sổ nằm trong thẻ nhà hàng.",
              "Tỷ lệ = NEG / toàn bộ nhận xét tháng. Cơ sở = trung bình cộng tỷ lệ của 2 hoặc 3 tháng lịch liền trước.",
              coverage_text(p),
              threshold_text(p),
              "Cần đạt cả ba điều kiện, xét đỏ trước vàng. Không gộp kết quả hai cửa sổ để tạo màu mới.",
              "Tháng trống không có tỷ lệ 0; tháng chưa kết thúc không đánh giá. Thiếu nhãn máy thống nhất: chờ dự đoán.",
              "Kết quả ưu tiên nhãn tay xác nhận, còn lại dùng nhãn máy; chưa phải đánh giá độc lập độ chính xác cảnh báo.",
              "", "Mỗi bảng có baseline_months để phân biệt hai cửa sổ. Số nhà hàng và nhà hàng–tháng tổng đã bỏ trùng.",
              "Kết quả lưu tại thư mục lần chạy; hướng dẫn nằm tại docs/he_thong/tang2/HUONG_DAN_TANG_2.md."]
    return "\n".join(lines) + "\n"


def output_root(cfg):
    return alert_output_dir(cfg, 2)


def signature(cfg, inputs, policy, cutoff):
    return {"version": VERSION, "policy": policy, "as_of": cutoff.isoformat(),
            "system_manifest_sha256": sha(inputs["paths"]["manifest"]),
            "master_sha256": sha(inputs["paths"]["master"]),
            "system_outputs_sha256": inputs["manifest"]["output_sha256"],
            "builder_sha256": sha(__file__), "view_sha256": sha(Path(__file__).with_name("view_trend_alerts.py")),
            "system_reader_sha256": sha(cfg["_root"] / "src/pipeline/prepare_system_data.py"),
            "update_reader_sha256": sha(cfg["_root"] / "src/pipeline/update_system_data.py")}


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".building.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("Một tiến trình đang tạo tầng 2. Nếu tiến trình đã tắt, xóa " + str(lock)) from exc
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def report_base(project_root):
    return Path(project_root) / "reports/alerts/foody_v01/tang2"


def report_directory_from_manifest(manifest, project_root=ROOT):
    relative = Path(manifest["report_directory"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Đường dẫn báo cáo không hợp lệ.")
    target = (Path(project_root) / relative).resolve()
    require(target.is_relative_to(report_base(project_root).resolve()), "Báo cáo phải nằm trong reports/alerts/foody_v01/tang2.")
    return target


def current_directory(root):
    root = Path(root)
    pointer = read_json(root / "current_trend.json")
    require(pointer.get("layout_version") == LAYOUT_VERSION,
            "Kết quả còn ở bố cục cũ; chạy ô cập nhật System rồi xây dựng lại tầng 2.")
    name = pointer.get("directory_name", "")
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            and name not in {".", ".."}, "Tên thư mục lần chạy không hợp lệ.")
    directory = root / "lan_chay" / name
    require(sha(directory / "trend_manifest.json") == pointer["manifest_sha256"],
            "Kết quả tầng 2 không khớp bản đang dùng.")
    return directory


def verify_run(directory, expected_signature=None, project_root=ROOT):
    directory = Path(directory)
    manifest = read_json(directory / "trend_manifest.json")
    require(manifest.get("status") == "complete" and manifest.get("layout_version") == LAYOUT_VERSION,
            "Kết quả tầng 2 chưa hoàn tất hoặc còn bố cục cũ; chạy lại ô tạo kết quả.")
    if expected_signature is not None:
        require(manifest["signature"] == expected_signature, "Tầng 2 chưa khớp dữ liệu/cấu hình hiện tại; chạy lại ô tạo kết quả.")
    for name, digest in manifest["output_sha256"].items():
        require(Path(name).name == name and sha(directory / name) == digest, "Dữ liệu tầng 2 đã đổi hoặc thiếu: " + name)
    reports = report_directory_from_manifest(manifest, project_root)
    for name, digest in manifest["report_sha256"].items():
        require(Path(name).name == name and sha(reports / name) == digest, "Báo cáo tầng 2 đã đổi hoặc thiếu: " + name)
    return manifest


def run(cfg, profile=None, as_of=None, inputs_override=None, verify_only=False):
    policy = policy_for(cfg, profile)
    cutoff = as_of_time(as_of or datetime.now(ZoneInfo(policy['timezone'])).date().isoformat(), policy["timezone"])
    inputs = inputs_override or load_system_inputs(cfg["_root"], view="operational", config=cfg["_config"],
                                                 snapshot_dir=cfg.get("_system_snapshot"))
    cfg = dict(cfg, _system_snapshot=inputs["paths"]["directory"])
    identity = signature(cfg, inputs, policy, cutoff)
    run_id = hashlib.sha256(compact(identity).encode()).hexdigest()[:24]
    root = output_root(cfg)
    with output_lock(root):
        pointer_path = root / "current_trend.json"
        current = read_json(pointer_path) if pointer_path.exists() else {}
        # Đầu vào không đổi: dùng lại chính lần hiện hành, không tạo thư mục mới mỗi lần mở.
        if current.get("run_id") == run_id and current.get("layout_version") == LAYOUT_VERSION:
            destination = current_directory(root)
            manifest = verify_run(destination, identity, cfg["_root"])
            return {**manifest, "directory": str(destination),
                    "reports_directory": str(report_directory_from_manifest(manifest, cfg["_root"]))}
        require(not verify_only, "Chưa có kết quả hiện hành khớp đầu vào đang dùng.")
        result = aggregate(inputs, policy, cutoff)
        sequence = 1
        name = f"canh_bao_v{sequence:02d}"
        history = root / "lan_chay"
        history.mkdir(exist_ok=True)
        destination = history / name
        while destination.exists():
            sequence += 1
            name = f"canh_bao_v{sequence:02d}"
            destination = history / name
        staging = Path(tempfile.mkdtemp(prefix=".dang_tao_", dir=history))
        report_staging = staging
        report_destination = destination
        data_published = False
        try:
            write_csv(staging / "trend_coverage.csv", COVERAGE_FIELDS, result["coverage"])
            write_csv(staging / "trend_metrics.csv", METRIC_FIELDS, result["metrics"])
            write_csv(staging / "trend_alerts.csv", METRIC_FIELDS, result["alerts"])
            save_json(staging / "trend_summary.json", result["summary"])
            (report_staging / "bao_cao_tang_2.txt").write_text(report_text(result), encoding="utf-8")
            from src.alerts.tang2.view_trend_alerts import render_dashboard
            (report_staging / "demo_tang_2.html").write_text(render_dashboard(result), encoding="utf-8")
            require(signature(cfg, inputs, policy_for(cfg, profile), cutoff) == identity,
                    "Đầu vào hoặc cấu hình đổi khi đang tạo tầng 2; giữ kết quả trước đó.")
            files = ["trend_coverage.csv", "trend_metrics.csv", "trend_alerts.csv", "trend_summary.json"]
            manifest = {"version": VERSION, "layout_version": LAYOUT_VERSION, "status": "complete",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "run_id": run_id, "directory_name": name, "signature": identity, "summary": result["summary"],
                        "output_sha256": {file: sha(staging/file) for file in files},
                        "report_directory": report_destination.relative_to(cfg["_root"]).as_posix(),
                        "report_sha256": {file: sha(report_staging/file) for file in ("bao_cao_tang_2.txt", "demo_tang_2.html")},
                        "recalculation_mode": "full_current_snapshot", "model_retrained": False, "predictions_generated": False}
            save_json(staging / "trend_manifest.json", manifest)
            # Kích hoạt con trỏ sau khi cả dữ liệu và báo cáo đều đã hoàn tất.
            staging.rename(destination)
            data_published = True
            verify_run(destination, identity, cfg["_root"])
            temporary = root / "current_trend.json.tmp"
            save_json(temporary, {"layout_version": LAYOUT_VERSION, "run_id": run_id, "directory_name": name,
                                  "profile": policy["profile"], "manifest_sha256": sha(destination / "trend_manifest.json")})
            temporary.replace(pointer_path)
        except BaseException:
            # Đây chỉ là thư mục mới của lần thất bại, không chạm bản đang hoạt động.
            activated = pointer_path.exists() and read_json(pointer_path).get("directory_name") == name
            if not activated:
                if data_published:
                    shutil.rmtree(destination, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return {**manifest, "directory": str(destination), "reports_directory": str(report_destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--as-of", default=None, help="Ngày YYYY-MM-DD, mặc định hết ngày hiện tại tại Việt Nam")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(load_settings(args.config, args.project_root), args.profile, args.as_of, verify_only=args.verify_only)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("Dữ liệu tầng 2:", result["directory"])
    print("Báo cáo và HTML:", result["reports_directory"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, OSError, json.JSONDecodeError) as exc:
        print("DỪNG:", exc, file=sys.stderr)
        raise SystemExit(2)
