"""Đề xuất bằng chứng nguyên văn theo mệnh đề và chấm mức vấn đề của cả nhận xét.

Đây là bộ luật hỗ trợ đọc, không phải mô hình đã được đo độ chính xác ngữ nghĩa.
Không sửa nhãn, loại sự cố hoặc bằng chứng nguồn; không cần gọi lại mô hình.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import lru_cache
import re
import unicodedata

MIN_WORDS = 4
MIN_LETTERS = 15
MAX_WORDS = 45
MAX_CHARS = 320
MAX_SPANS = 2
WORD = re.compile(r"\w+(?:['’’-]\w+)*", re.UNICODE)

# Từ gợi khía cạnh và biểu đạt cảm xúc. Các luật âm đang dùng trong cấu hình
# system được ghép thêm ở EvidenceSelector; không đổi bộ luật A8 đã chốt.
PROFILES = {
    "A1": {
        "terms": "nhân viên|phục vụ|thái độ|chủ quán|bảo vệ|nhân sự|tiếp khách|khách hàng",
        "pos": "thân thiện|nhiệt tình|dễ thương|lịch sự|niềm nở|vui vẻ|chu đáo|tận tình|nhanh nhẹn|tốt|ổn",
        "neg": "khó chịu|cộc cằn|thờ ơ|bất lịch sự|hỗn|cau có|hời hợt|khinh|coi thường|mất dạy|chửi|mắng|xúc phạm|tệ|kém|ngáo|không nhiệt tình",
    },
    "A2": {
        "terms": "chờ|đợi|lên món|ra món|giao món|tốc độ|nhanh nhẹn|phục vụ|nhân viên|thời gian|nước uống",
        "pos": "nhanh|nhanh nhẹn|mau|kịp thời|đúng giờ|nhanh chóng",
        "neg": "chậm|lâu|đợi mãi|chờ mãi|ăn hết mùa|bỏ về|chưa lên|chưa có|không được phục vụ",
    },
    "A3": {
        "terms": "món ăn|món|đồ ăn|thức ăn|ăn|uống|khẩu vị|khẩu phần|nêm|mì|mỳ|bún|phở|cơm|thịt|cá|gà|bò|rau|sườn|tôm|mực|bánh|ram|kimbap|sữa|trà|cà phê|nước|hải sản",
        "pos": "ngon|ngon miệng|ổn|tươi|đậm đà|vừa miệng|hợp khẩu vị|thơm|hấp dẫn|xuất sắc|ok|ăn được|uống được|tạm được|thấm|hết sẩy",
        "neg": "dở|tệ|mặn|nhạt|khô|dai|tanh|nguội|khét|ôi thiu|bị hỏng|mốc|hôi|chưa chín|còn sống|ít|ngán|quá ngọt|ngọt như đường|ngọt hơn chè|nhão|toàn bột|nhiều bột|dính mực|không hợp|không giống|không phải|ko phải|quán nên",
    },
    "A4": {
        "terms": "không gian|quán|phòng|chỗ ngồi|bàn ghế|máy lạnh|điều hòa|điều hoà|quạt|tầng|đường phố|view|vị trí|chỗ đậu|bãi xe|cơ sở vật chất|nhìn ra",
        "pos": "thoáng|rộng|rộng rãi|mát|thoải mái|đẹp|hay|thích|ấm cúng|yên tĩnh|sáng sủa|dễ tìm",
        "neg": "chật|nóng|ồn|bí|tối|ngộp|hỏng|cũ|lung lay|trơn|chập điện|rò điện|sập|khó tìm|âm u",
    },
    "A5": {
        "terms": "vệ sinh|sạch|bẩn|dơ|hôi|ruồi|gián|sâu|tóc|giòi|dị vật|đau bụng|tiêu chảy|ngộ độc|nôn|ói|nhập viện|cấp cứu|khó thở|ăn xong|toilet|nhà vệ sinh|bát|đũa|chén|ly|cốc",
        "pos": "sạch|sạch sẽ|hợp vệ sinh|gọn gàng|đảm bảo vệ sinh",
        "neg": "bẩn|dơ|hôi|ruồi|gián|sâu|tóc|giòi|dị vật|đau bụng|tiêu chảy|ngộ độc|nôn|ói|nhập viện|cấp cứu|khó thở|ôi thiu|mốc|hết hạn",
    },
    "A6": {
        "terms": "giá|tiền|bill|hóa đơn|hoá đơn|mắc|đắt|rẻ|phụ thu|đồng tiền|chi phí|thanh toán",
        "pos": "rẻ|hợp lý|hợp lí|phải chăng|vừa túi tiền|xứng đáng|đáng tiền|phù hợp với sinh viên|bình dân|ổn",
        "neg": "đắt|mắc|cao|chát|chặt chém|tính khống|thu khống|không xứng|không đáng|phụ thu|bất hợp lý|bất hợp lí|tính sai",
    },
    "A7": {
        "terms": "phản hồi|trả lời|khiếu nại|phản ánh|góp ý|giải quyết|xử lý|xử lí|hỗ trợ|xin lỗi|hoàn tiền|bồi thường|đổi món|tin nhắn",
        "pos": "kịp thời|tận tình|nhanh|thỏa đáng|thoả đáng|tốt|hợp lý|hợp lí|xin lỗi|hoàn tiền|đổi món",
        "neg": "chậm|kém|qua loa|từ chối|bỏ qua|không chịu|không giải quyết|không xử lý|không trả lời|không phản hồi|không hoàn tiền|im lặng",
    },
}

INCIDENT_TERMS = {
    "SUC_KHOE": "đau bụng|tiêu chảy|ngộ độc|nôn|ói|nhập viện|cấp cứu|khó thở|dị ứng|môi tê|tê môi|tay chân bủn rùn|chóng mặt|bị sốt|sốt cao|mệt",
    "PHAN_BIET_DOI_XU": "phân biệt|coi thường|khinh|khách tây|khách việt|người việt|người nước ngoài|khách nước ngoài|kỳ thị|kì thị",
    "XUC_PHAM_NGHIEM_TRONG": "chửi|mắng|xúc phạm|lăng mạ|sỉ nhục|mất dạy|nhục mạ|chửi bới",
    "AN_TOAN_TAI_CO_SO": "cháy|bỏng|điện giật|sập|ngã|té|bị thương|chảy máu|nổ|trơn|vỡ|rò điện",
    "TRAI_NGHIEM_CUC_TIEU_CUC": "tệ nhất|kinh khủng|không thể chấp nhận|thảm họa|quá tệ|không quay lại|thất vọng|tồi tệ",
    "BAO_LUC_DE_DOA": "đánh|đe dọa|đe doạ|hăm dọa|hăm doạ|dọa|doạ|hành hung|tát|đập|giết|dao",
}
STOP_ONLY = {"tớ", "tôi", "mình", "ăn", "uống", "cơ", "mà", "và", "thì", "là", "cũng", "của", "nữa", "nè", "đó", "này"}
NEGATION = re.compile(r"(?:không|ko|k|chưa|chẳng|chả|đéo)(?:\s+(?:hề|được|thật|sự|có|bị|thấy|quá|rất)){0,2}\s*$", re.I)


def normalized(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).casefold()).strip()


def words(value):
    return list(WORD.finditer(value))


def phrase_pattern(terms):
    entries = sorted(set(terms.split("|")), key=len, reverse=True)
    body = "|".join(re.escape(x).replace(r"\ ", r"\s+") for x in entries if x)
    return re.compile(r"(?<!\w)(?:" + (body or r"(?!)") + r")(?!\w)", re.I)


def score_review(labels, has_a8=False):
    """Các mốc chính xác 1/3, 2/3; A8 không nhận mức độ/điểm hiển thị."""
    counts = Counter(labels) if not isinstance(labels, Counter) else labels
    if any(k not in {"NONE", "POS", "NEU", "NEG", "MIX"} for k in counts):
        raise ValueError("Có nhãn ngoài bộ A1–A7.")
    mentioned = sum(counts[x] for x in ("POS", "NEU", "NEG", "MIX"))
    if has_a8 or not mentioned or not (counts["NEG"] or counts["MIX"]):
        return {"level": None, "score": None, "mentioned": mentioned}
    score = Fraction(2 * counts["NEG"] + counts["MIX"], 2 * mentioned)
    level = "nhe" if score < Fraction(1, 3) else "vua" if score < Fraction(2, 3) else "nang"
    return {"level": level, "score": float(score), "mentioned": mentioned}


def trimmed(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


@lru_cache(maxsize=128)
def clauses(text):
    """Tách bằng vị trí trong văn bản gốc, giữ dấu và không sửa chính tả."""
    ranges, start = [], 0
    # Không cắt dấu chấm nằm giữa hai chữ số của giá hoặc số thập phân.
    for match in re.finditer(r"[.!?;]+|\n+", text):
        if (match.group() == "." and match.start() > 0 and match.end() < len(text)
                and text[match.start()-1].isdigit() and text[match.end()].isdigit()):
            continue
        end = match.end()
        ranges.append((start, end))
        start = end
    ranges.append((start, len(text)))
    result = []
    splitter = re.compile(
        r"(?<!\w)(?i:nhưng|tuy nhiên|cơ mà|bù lại|cho tới khi|cho đến khi|đến khi)(?!\w)|"
        r"(?<=,)\s*(?=(?i:giá|nhân viên|phục vụ|không gian|vệ sinh|nêm nếm)\b)|"
        r"\s+(?=(?:Tớ cũng|Mình thấy|Tôi thấy|Nhân viên|Không gian|Giá cả|Còn tầng|Từ tầng)\b)", re.UNICODE)
    for a, b in ranges:
        cuts = [a]
        for match in splitter.finditer(text, a, b):
            cut = match.start()
            if cut > cuts[-1] and WORD.search(text[cuts[-1]:cut]):
                cuts.append(cut)
        cuts.append(b)
        for left, right in zip(cuts, cuts[1:]):
            span = trimmed(text, left, right)
            if span:
                result.append(span)
    return tuple(result)


def clipped(text, start, end, focus=None):
    """Giới hạn đoạn đọc nhanh; chỉ cắt ở biên từ, không cắt trong từ."""
    ws = list(WORD.finditer(text, start, end))
    if not ws:
        return start, end
    if len(ws) <= MAX_WORDS and end - start <= MAX_CHARS:
        return start, end
    center = next((i for i, w in enumerate(ws) if focus is not None and w.end() > focus), len(ws) // 2)
    left = max(0, min(center - 12, len(ws) - MAX_WORDS))
    right = min(len(ws), left + MAX_WORDS)
    while right - left > MIN_WORDS and ws[right - 1].end() - ws[left].start() > MAX_CHARS:
        if center - left > right - 1 - center:
            left += 1
        else:
            right -= 1
    return ws[left].start(), ws[right - 1].end()


class EvidenceSelector:
    """Giữ đoạn máy đủ ý; lọc mảnh rời rồi bổ sung ngữ cảnh theo luật."""

    def __init__(self, policy=None):
        self.compiled = {a: {k: phrase_pattern(v) for k, v in profile.items()}
                         for a, profile in PROFILES.items()}
        self.rule_patterns = {a: [] for a in PROFILES}
        for rule in (policy or {}).get("aspect_rules", []):
            for pattern in rule.get("patterns_any", []):
                self.rule_patterns[rule["aspect"]].append(re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", re.I))
        self.incidents = {code: phrase_pattern(terms) for code, terms in INCIDENT_TERMS.items()}

    def polarity(self, value, aspect):
        pos, neg = 0, 0
        for kind in ("pos", "neg"):
            for match in self.compiled[aspect][kind].finditer(value):
                if kind == "pos" and normalized(match.group()) == "thơm" and re.search(r"trái\s*$", value[:match.start()], re.I):
                    continue
                before = value[max(0, match.start() - 24):match.start()]
                inverted = bool(NEGATION.search(before)) and not re.search(r"không\s+(?:chỉ|những)\s*$", before, re.I)
                if (kind == "pos") != inverted:
                    pos += 1
                else:
                    neg += 1
        # Các luật hiện có chỉ bổ sung gợi ý ngữ cảnh âm, không gán lại mức độ/A8.
        if any(not NEGATION.search(value[max(0, m.start()-24):m.start()])
               for p in self.rule_patterns[aspect] for m in p.finditer(value)):
            neg += 1
        return pos, neg

    def short_clear(self, value, aspect, label):
        """Ngoại lệ độ dài phải là cụm đánh giá rõ nghĩa, không chỉ chứa một từ khóa."""
        value = normalized(value).strip(" .,!?:;\"'()[]{}…")
        if not value or set(WORD.findall(value)) <= STOP_ONLY:
            return False
        profile = PROFILES[aspect]
        # Cho phép chủ thể ngắn: 'món ngon', 'giá quá đắt', 'nhân viên dễ thương'.
        subjects = sorted(profile["terms"].split("|"), key=len, reverse=True)
        for subject in subjects:
            if value.startswith(subject + " "):
                value = value[len(subject):].strip()
                break
        value = re.sub(r"^(?:(?:rất|quá|khá|hơi|cực|siêu|thật|thực sự)\s+)+", "", value)
        value = re.sub(r"\s+(?:lắm|quá|ghê|thật|luôn)$", "", value)
        terms = profile["pos"].split("|") + profile["neg"].split("|")
        if value not in terms:
            # Phủ định rõ nghĩa cũng được giữ nguyên, ví dụ 'không ngon'.
            value_without_negation = re.sub(r"^(?:không|ko|k|chưa|chẳng|chả)\s+", "", value)
            if value_without_negation not in terms or value_without_negation == value:
                return False
        p, n = self.polarity(value, aspect)
        return bool((label == "POS" and p and not n) or (label == "NEG" and n and not p)
                    or (label == "MIX" and (p or n)))

    def qualifies(self, value, aspect, label):
        length_ok = len(words(value)) >= MIN_WORDS and sum(c.isalnum() for c in value) >= MIN_LETTERS
        return length_ok or self.short_clear(value, aspect, label)

    def _span(self, text, start, end, polarity="NEU"):
        return {"start": start, "end": end, "text": text[start:end], "polarity": polarity}

    def propose(self, text, aspect, label, manual_spans=None, model_spans=None):
        if aspect not in PROFILES or label not in {"NONE", "POS", "NEU", "NEG", "MIX"}:
            raise ValueError("Khía cạnh hoặc nhãn không hợp lệ.")
        if label == "NONE":
            return {"kind": "none", "spans": [], "complete": True}
        # Ưu tiên bằng chứng tay đã đủ ý; không mở rộng một đoạn chuẩn sang câu khác.
        # Vẫn chặn mảnh quá ngắn và giữ nguyên bản nguồn trong system.
        if manual_spans and label != "MIX":
            preserved = []
            for span in manual_spans:
                a, b = span["start"], span["end"]
                if not (type(a) is int and type(b) is int and 0 <= a < b <= len(text)
                        and text[a:b] == span["text"]):
                    raise ValueError("Bằng chứng gán tay không khớp nhận xét gốc.")
                value = text[a:b]
                if (len(words(value)) <= MAX_WORDS and len(value) <= MAX_CHARS
                        and self.qualifies(value, aspect, label)):
                    preserved.append(self._span(text, a, b, label))
            if preserved:
                return {"kind": "evidence", "spans": preserved[:MAX_SPANS], "complete": True}
        machine = []
        for span in model_spans or []:
            a, b = span["start"], span["end"]
            if not (type(a) is int and type(b) is int and 0 <= a < b <= len(text) and text[a:b] == span["text"]):
                raise ValueError("Bằng chứng máy không khớp nhận xét gốc.")
            inside_word = ((a > 0 and text[a-1].isalnum() and text[a].isalnum()) or
                           (b < len(text) and text[b-1].isalnum() and text[b].isalnum()))
            value = text[a:b]
            if (not inside_word and len(words(value)) <= MAX_WORDS and len(value) <= MAX_CHARS
                    and self.qualifies(value, aspect, label)):
                machine.append(span)
        # Không bắt bộ trích chỉ được diễn đạt bằng từ có trong từ điển.
        # Đây vẫn là đề xuất máy; không chuyển thành xác nhận tay.
        if machine and label != "MIX" and not manual_spans:
            return {"kind": "evidence", "spans": [self._span(text, s["start"], s["end"], label)
                                                     for s in machine[:MAX_SPANS]], "complete": True}
        candidates = []
        for a, b in clauses(text):
            value = text[a:b]
            hits = list(self.compiled[aspect]["terms"].finditer(value))
            has_manual = any(a < s["end"] and b > s["start"] for s in manual_spans or [])
            has_model = any(a < s["end"] and b > s["start"] for s in machine)
            if not hits and not self.short_clear(value, aspect, label) and not has_manual and not has_model:
                continue
            focus = a + hits[0].start() if hits else a
            x, y = clipped(text, a, b, focus)
            value = text[x:y]
            if not self.qualifies(value, aspect, label):
                continue
            p, n = self.polarity(value, aspect)
            score = min(len(hits), 3) + 3 * min(p + n, 3) + 2 * has_manual + 2 * has_model - len(words(value)) / 80
            candidates.append({"start": x, "end": y, "text": value, "p": p, "n": n,
                               "score": score, "manual": has_manual})
        candidates.sort(key=lambda c: (-c["score"], c["start"]))
        selected = []
        if label == "MIX":
            for kind in ("p", "n"):
                found = next((c for c in candidates if c[kind]), None)
                if found and not any(c["start"] == found["start"] and c["end"] == found["end"] for c in selected):
                    selected.append(found)
            complete = any(c["p"] for c in selected) and any(c["n"] for c in selected)
        else:
            appropriate = lambda c: (label == "NEU" or c["manual"] or
                label == "POS" and c["p"] and not c["n"] or
                label == "NEG" and c["n"] and not c["p"])
            selected = [c for c in candidates if appropriate(c)][:MAX_SPANS]
            complete = bool(selected)
        if selected and complete:
            spans = [self._span(text, c["start"], c["end"], "MIX" if c["p"] and c["n"] else
                               "POS" if c["p"] else "NEG" if c["n"] else "NEU") for c in selected]
            return {"kind": "evidence", "spans": sorted(spans, key=lambda s: s["start"]), "complete": True}
        # Không gọi một đoạn chỉ khen là bằng chứng đầy đủ cho nhãn MIX.
        related = selected or candidates[:1]
        if related:
            return {"kind": "context", "spans": [self._span(text, c["start"], c["end"]) for c in related[:MAX_SPANS]], "complete": False}
        return {"kind": "context", "spans": self.context(text, aspect, preferred_spans=machine), "complete": False}

    def context(self, text, aspect=None, preferred_spans=None):
        """Ngữ cảnh xem nhanh là trích đoạn, không tự nhận là bằng chứng đã xác nhận."""
        ranked = []
        for a, b in clauses(text):
            value = text[a:b]
            hits = list(self.compiled[aspect]["terms"].finditer(value)) if aspect else []
            overlap = sum(a < s["end"] and b > s["start"] for s in preferred_spans or [])
            x, y = clipped(text, a, b, a + hits[0].start() if hits else a)
            value = text[x:y]
            if len(words(value)) < MIN_WORDS or sum(c.isalnum() for c in value) < MIN_LETTERS:
                if hits:
                    # Cụm nhắc khía cạnh quá ngắn: mở ngữ cảnh quanh chính cụm đó,
                    # thay vì chọn một câu đủ dài nhưng nói về khía cạnh khác.
                    x, y = clipped(text, 0, len(text), a + hits[0].start())
                    value = text[x:y]
                    if len(words(value)) >= MIN_WORDS and sum(c.isalnum() for c in value) >= MIN_LETTERS:
                        ranked.append((len(hits) + overlap * 3, -(y-x), x, y))
                continue
            ranked.append((len(hits) + overlap * 3, -(y - x), x, y))
        if ranked:
            _, _, a, b = max(ranked)
            return [self._span(text, a, b)]
        # Nhận xét rất ngắn: giữ nguyên ngữ cảnh ngắn, không cắt thành các từ rời.
        a, b = clipped(text, 0, len(text), 0)
        return [self._span(text, a, b)] if a < b else []

    def propose_incident(self, text, incident_types, original_spans):
        """Mã loại sự cố chỉ được đọc từ dữ liệu; từ khóa chỉ giúp chọn đoạn giải thích."""
        codes = [c.strip() for c in incident_types.split("|") if c.strip()]
        selected = []
        for code in codes:
            pattern = self.incidents.get(code)
            ranked = []
            for a, b in clauses(text):
                value = text[a:b]
                matches = [m for m in pattern.finditer(value)
                           if not NEGATION.search(value[max(0, m.start()-24):m.start()])] if pattern else []
                overlap = sum(a < s["end"] and b > s["start"] for s in original_spans)
                if not matches:
                    continue
                x, y = clipped(text, a, b, a + matches[0].start())
                value = text[x:y]
                # Sự cố cực ngắn chỉ chấp nhận khi toàn cụm là biểu đạt rõ của loại đã lưu.
                short_exact = bool(pattern and pattern.fullmatch(normalized(value).strip(" .,!?:;")))
                if not (len(words(value)) >= MIN_WORDS and sum(c.isalnum() for c in value) >= MIN_LETTERS) and not short_exact:
                    continue
                ranked.append((len(matches) * 4 + overlap * 2, -(y - x), x, y))
            if ranked:
                _, _, a, b = max(ranked)
                if not any(s["start"] == a and s["end"] == b for s in selected):
                    selected.append(self._span(text, a, b, "NEG"))
        if selected:
            return {"kind": "evidence", "spans": sorted(selected, key=lambda s: s["start"]), "complete": True}
        return {"kind": "context", "spans": self.context(text, preferred_spans=original_spans), "complete": False}
