"""Nối tầng 2 với bộ cập nhật chung đã có, không tạo bản sao bộ cập nhật."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import stat
import tempfile
import hashlib

MODULE = "src.alerts.tang2.build_trend_alerts"
CONFIG = "configs/alerts/tang2/trend_alerts.json"
KNOWN_MODULES = {MODULE, "src.alerts_tang2.build_tier2_alerts", "src.alerts.build_tier2_alerts", "src.alerts.build_trend_alerts", "src.alerts.tang2.tao"}
KNOWN_CONFIGS = {CONFIG, "configs/alerts_tang2/tier2_alerts.json", "configs/alerts/tier2_alerts.json", "configs/alerts/tang_2.json", "configs/alerts/tang2.json"}
LEGACY_UPDATER_SHA256 = "f41d18f479aa3cbef480a847ff8d0401c0a67ad00b9eadeda1f8da8083a870e5"


def _add_legacy_hook(source):
    """Thêm bước gọi tầng 2 vào đúng bộ cập nhật gốc đã xác định, không thay bước tầng 1."""
    windows_lines = "\r\n" in source
    normalized = source.replace("\r\n", "\n")
    tree = ast.parse(normalized)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_batch")
    lines = normalized.splitlines(keepends=True)
    body = "".join(lines[function.lineno-1:function.end_lineno])
    head = '    base = location(cfg, "system_data_dir")\n'
    tail = '            atomic_json(alert_manifest_path, alert_manifest)\n'
    if body.count(head) != 1 or body.count(tail) != 1 or body.count('"tier2_built": False') != 2:
        raise ValueError("Bộ cập nhật gốc khác cấu trúc đã xác định; giữ nguyên tệp.")
    body = body.replace(head, head + '    tier2_config = cfg["_root"] / "' + CONFIG + '"\n'
                        '    tier2_enabled = tier2_config.is_file() and jread(tier2_config).get("enabled") is True\n')
    body = body.replace('"tier2_built": False', '"tier2_built": tier2_enabled')
    body = body.replace(tail, tail + '            if tier2_enabled:\n'
                        '                from ' + MODULE + ' import run as build_trend\n'
                        '                trend_inputs = dict(staged_inputs, aspects=csv_rows(staging / cfg["system_aspects_file"]))\n'
                        '                build_trend(staged_cfg, inputs_override=trend_inputs)\n')
    result = "".join(lines[:function.lineno-1]) + body + "".join(lines[function.end_lineno:])
    return result.replace("\n", "\r\n") if windows_lines else result

def archive_previous_tang2(project_root):
    """Tương thích sổ chạy cũ: đã tắt việc tạo archive hoặc di chuyển tệp tự động."""
    return {"archived": [], "not_moved": [], "archive_dir": None, "disabled": True}


def connect_shared_updater(project_root):
    root = Path(project_root).resolve()
    path = root / "src/pipeline/update_system_data.py"
    if not path.is_file():
        raise FileNotFoundError("Thiếu bộ cập nhật chung đã có: " + str(path)
                                + ". Gói tầng 2 không tạo thêm một bộ cập nhật chung.")
    before = path.read_bytes()
    source = before.decode("utf-8-sig")
    if "tier2_config" not in source:
        digest = hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()
        if digest == LEGACY_UPDATER_SHA256:
            source = _add_legacy_hook(source)
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module in KNOWN_MODULES and any(a.name == "run" for a in n.names)]
    if not imports or "tier2_config" not in source or "tier2_enabled" not in source:
        raise ValueError("Bộ cập nhật chung chưa có liên kết tầng 2 theo bản đã gửi; "
                         "giữ nguyên tệp hiện tại và dùng bộ cập nhật chung trong bản gốc của dự án.")
    changed = source
    for node in imports:
        changed = changed.replace("from " + node.module + " import", "from " + MODULE + " import")
    for value in KNOWN_CONFIGS:
        changed = changed.replace(value, CONFIG)
    changed = changed.replace('"configuration_source": "alert_policy.trend"',
                              '"configuration_source": "' + CONFIG + '"')
    # Cập nhật hợp đồng đầu vào, cả hai cửa sổ đều tính lại từ toàn bản system.
    changed = changed.replace('"baseline_previous_calendar_periods": 3,',
                              '"baseline_windows": [2, 3], "recalculation_mode": "full_current_snapshot",')
    checked = ast.parse(changed)
    if any(isinstance(n, ast.ImportFrom) and n.module in KNOWN_MODULES - {MODULE}
           for n in ast.walk(checked)):
        raise ValueError("Chưa đổi được đúng lệnh gọi tầng 2; giữ nguyên bộ cập nhật chung.")
    encoded = changed.encode("utf-8")
    if before.startswith(b"\xef\xbb\xbf"):
        encoded = b"\xef\xbb\xbf" + encoded
    if encoded == before:
        return {"shared_path": str(path), "changed": False, "shared_copies_created": 0}
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary = tempfile.mkstemp(prefix=".tang2_link_", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        if path.read_bytes() != before:
            raise ValueError("Bộ cập nhật chung đang được sửa bởi tiến trình khác; chưa đổi liên kết.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"shared_path": str(path), "changed": True, "shared_copies_created": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    print(json.dumps(connect_shared_updater(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
