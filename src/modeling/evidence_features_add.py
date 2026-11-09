from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from pathlib import Path
from .evidence_core_add import (encode_bio, make_windows, read_jsonl, sha256, text_hash, write_json)


def alignment_char(ch):
    # Một số bản cầu nối Java trả ký hiệu ngoài BMP thành mã thấp 16 bit.
    # Chỉ dùng biểu diễn này để căn vị trí; văn bản/đoạn đầu ra vẫn là bản gốc.
    if ord(ch) > 0xFFFF:
        ch = chr(ord(ch) & 0xFFFF)
    ch = {"“": '"', "”": '"', "‘": "'", "’": "'", "…": "...",
          "–": "-", "—": "-", "‐": "-"}.get(ch, ch)
    return "".join(c for c in unicodedata.normalize("NFD", ch.casefold())
                   if unicodedata.category(c) != "Mn" and not c.isspace() and c != "_")


def compact(text):
    chars, positions = [], []
    for i, ch in enumerate(text):
        normalized = alignment_char(ch)
        chars.extend(normalized)
        positions.extend([i] * len(normalized))
    return "".join(chars), positions


def ignored_gap(text):
    return all(c.isspace() or c == "_" or unicodedata.category(c)[0] in "PSC" for c in text)


def align_words(text, words):
    """Ánh xạ đơn điệu, chấp nhận chuẩn hóa dấu thanh nhưng giữ tọa độ gốc.

    Không tìm gần đúng hoặc tự bỏ qua chữ/số. Có thể bỏ ký hiệu mà bộ tách từ bỏ.
    """
    raw, positions = compact(text)
    cursor, result, skipped = 0, [], 0
    for word in words:
        norm, _ = compact(word)
        if not norm:
            continue
        start = raw.find(norm, cursor)
        if start < 0:
            raise ValueError(f"Không ánh xạ được từ đã tách: {word!r}")
        if start > cursor:
            gap_start = positions[cursor] if cursor < len(positions) else len(text)
            gap_end = positions[start]
            if not ignored_gap(text[gap_start:gap_end]):
                raise ValueError(f"Bộ tách từ làm mất chữ/số trước {word!r}")
            skipped += start - cursor
        result.append((word, positions[start:start+len(norm)]))
        cursor = start + len(norm)
    if cursor < len(raw):
        if not ignored_gap(text[positions[cursor]:]):
            raise ValueError("Bộ tách từ làm mất chữ/số ở cuối nhận xét")
        skipped += len(raw) - cursor
    if not result:
        raise ValueError("Bộ tách từ trả về văn bản rỗng")
    return result, skipped


class Segmenter:
    _engines = {}

    def __init__(self, directory, download=False):
        import py_vncorenlp
        # Một số máy có JRE nhưng pyjnius không tự tìm được vì không có javac.
        if not os.environ.get("JAVA_HOME") and shutil.which("java"):
            candidate = Path(shutil.which("java")).resolve().parent.parent
            if any((candidate/p).exists() for p in ["lib/server/libjvm.so", "bin/server/jvm.dll", "lib/server/libjvm.dylib"]):
                os.environ["JAVA_HOME"] = str(candidate)
        directory = Path(directory).resolve()
        if str(directory) in self._engines:
            self.directory = directory
            self.engine, self.identity = self._engines[str(directory)]
            return
        if not (directory / "VnCoreNLP-1.2.jar").exists():
            if not download:
                raise FileNotFoundError(f"Thiếu VnCoreNLP tại {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Tải tài nguyên tách từ vào {directory}", flush=True)
            py_vncorenlp.download_model(save_dir=str(directory))
        self.directory = directory
        previous = Path.cwd()
        try:
            self.engine = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(directory), max_heap_size="-Xmx2g")
        finally:
            os.chdir(previous)
        files = ["VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr"]
        self.identity = {p: sha256(directory / p) for p in files}
        self._engines[str(directory)] = (self.engine, self.identity)

    def words(self, text):
        return [w for sent in self.engine.word_segment(text) for w in sent.split()]


class FeatureBuilder:
    def __init__(self, tokenizer, segmenter, config):
        self.tokenizer, self.segmenter, self.config = tokenizer, segmenter, config
        self.prefixes = {}
        for aspect, description in config["aspects"].items():
            prompt = " ".join(segmenter.words(description))
            self.prefixes[aspect] = tokenizer.encode(prompt, add_special_tokens=False)

    def encode(self, review_id, text):
        words, skipped = align_words(text, self.segmenter.words(text))
        ids, offsets = [], []
        for word, raw_positions in words:
            pieces = self.tokenizer.tokenize(word)
            expected, _ = compact(word)
            cursor = 0
            for piece in pieces:
                spelling = piece[:-2] if piece.endswith("@@") else piece
                norm, _ = compact(spelling)
                end = cursor + len(norm)
                if end > len(raw_positions):
                    raise ValueError(f"BPE không khớp từ {word!r}")
                if expected[cursor:end] != norm:
                    raise ValueError(f"Nội dung BPE không khớp từ {word!r}")
                if norm:
                    start_char, end_char = raw_positions[cursor], raw_positions[end-1] + 1
                else:
                    start_char = end_char = raw_positions[max(cursor-1, 0)] + 1
                ids.append(self.tokenizer.convert_tokens_to_ids(piece))
                offsets.append([start_char, end_char])
                cursor = end
            if cursor != len(raw_positions):
                raise ValueError(f"BPE không bao phủ từ {word!r}")
        if not ids:
            raise ValueError("Không có token trong nhận xét")
        return {"review_id": review_id, "text_sha256": text_hash(text), "ids": ids,
                "offsets": offsets, "skipped_symbol_characters": skipped}

    def windows(self, review, pair):
        labels = encode_bio(review["offsets"], pair.get("spans", []))
        return make_windows(review["ids"], self.prefixes[pair["aspect"]], labels,
                            self.config["max_length"], self.config["overlap"],
                            self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)


def build_cache(path, reviews, builder, cache=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_name(path.stem + "_meta_add.json")
    vocab_hash = text_hash(json.dumps(sorted(builder.tokenizer.get_vocab().items()), ensure_ascii=False))
    identity = {"format": 1, "vocab_sha256": vocab_hash,
                "segmenter": builder.segmenter.identity, "alignment": "monotonic_unicode_v01"}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != identity:
            raise ValueError(f"Bộ nhớ đệm dùng cấu hình khác: {path}; đổi tên hoặc xóa tệp cache này")
    elif path.exists() and path.stat().st_size:
        raise ValueError(f"Cache thiếu thông tin nguồn: {path}")
    write_json(meta_path, identity)
    if cache is None:
        cache = {}
        if path.exists():
            with path.open("rb+") as existing:
                while True:
                    start = existing.tell()
                    line = existing.readline()
                    if not line:
                        break
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if existing.read(1):
                            raise ValueError(f"Cache hỏng ở giữa tệp: {path}")
                        existing.seek(start)
                        existing.truncate()
                        print("Đã bỏ dòng cache cuối bị ghi dở; sẽ tạo lại nhận xét đó.", flush=True)
                        break
                    cache[row["review_id"]] = row
                    if not line.endswith(b"\n"):
                        existing.seek(0, 2)
                        existing.write(b"\n")
                        break
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for i, row in enumerate(reviews):
            rid, text = row["review_id"], row["text"]
            if rid in cache:
                if cache[rid]["text_sha256"] != text_hash(text):
                    raise ValueError(f"Văn bản thay đổi so với cache: {rid}")
                continue
            try:
                encoded = builder.encode(rid, text)
            except ValueError as exc:
                raise ValueError(f"{rid}: {exc}") from exc
            cache[rid] = encoded
            f.write(json.dumps(encoded, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            if (i+1) % 100 == 0:
                print(f"Đã ánh xạ văn bản: {i+1}/{len(reviews)}", flush=True)
    return cache

