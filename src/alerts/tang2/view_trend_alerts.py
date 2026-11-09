"""Đọc kết quả tầng 2 và trình bày xu hướng của từng nhà hàng."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.alerts.tang2.build_trend_alerts import (
    ASPECT_NAMES, as_of_time, output_root, policy_for, current_directory, report_directory_from_manifest,
    require, signature, verify_run, month_number, month_name,
)
from src.modeling.phobert_system_common import load_settings
from src.pipeline.prepare_system_data import load_system_inputs


def pct(value):
    return '—' if value is None or value == '' else f'{100 * float(value):.1f}%'


def truth(value):
    return value is True or str(value).lower() in {'true', '1'}


def render_chart(rows, *, reference_rate=None):
    """Đường tỷ lệ tháng và cơ sở của đúng cửa sổ; ngắt đường tại tháng thiếu."""
    if not rows:
        return '<p>Chưa có chuỗi dữ liệu.</p>'
    by_month = {r['month']: r for r in rows}
    first, last = min(by_month), max(by_month)
    months = [month_name(n) for n in range(month_number(first), month_number(last) + 1)]
    width, height, left, top, bottom = 760, 220, 48, 15, 180
    xpos = lambda i: left + i * (width-left-18) / max(1, len(months)-1)
    ypos = lambda v: bottom - float(v) * (bottom-top)
    elements = []
    for tick in (0, .25, .5, .75, 1):
        y = ypos(tick)
        elements.append(f'<line x1="{left}" x2="{width-18}" y1="{y}" y2="{y}" stroke="#dde5ee"/><text x="3" y="{y+4}" font-size="11">{tick*100:g}%</text>')
    fields = [('complaint_rate', '#327fba')]
    if reference_rate is None:
        fields.append(('baseline_rate', '#c56a12'))
    else:
        y = ypos(reference_rate)
        elements.append(f'<line x1="{left}" x2="{width-18}" y1="{y}" y2="{y}" stroke="#cf8b37" stroke-width="2" stroke-dasharray="6 4"><title>Trung bình các tháng trước: {pct(reference_rate)}</title></line>')
    for field, color in fields:
        prev = None
        for i, month in enumerate(months):
            row = by_month.get(month, {})
            value = row.get(field)
            if field == 'baseline_rate' and not truth(row.get('eligible', False)):
                value = None
            if value is None or value == '':
                prev = None
                continue
            x, y = xpos(i), ypos(value)
            if prev:
                elements.append(f'<line x1="{prev[0]:.2f}" y1="{prev[1]:.2f}" x2="{x:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"/>')
            title = escape(f'{month}: {pct(value)}')
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"><title>{title}</title></circle>')
            prev = (x, y)
    step = max(1, (len(months)+5)//6)
    for i, month in enumerate(months):
        if i % step == 0 or i == len(months)-1:
            elements.append(f'<text x="{xpos(i):.2f}" y="205" text-anchor="middle" font-size="10">{month}</text>')
    legend = ('Nét xanh: tỷ lệ tiêu cực từng tháng. Nét cam đứt: mức trung bình dùng để so sánh.'
              if reference_rate is not None else 'Nét xanh: tỷ lệ tiêu cực từng tháng. Nét cam: mức trung bình của các tháng trước.')
    return (f'<svg class="trend" width="100%" viewBox="0 0 {width} {height}" role="img" aria-label="Tỷ lệ tiêu cực theo tháng và mức trung bình">'
            + ''.join(elements) + f'</svg><p class="muted">{legend}</p>')


def comparison_months(row):
    """Chỉ lấy các tháng thuộc phép so sánh đã tạo ra cảnh báo."""
    return [*json.loads(row['baseline_periods_json']), {
        'month': row['month'], 'review_count': row['review_count'],
        'negative_count': row['count_NEG'], 'complaint_rate': row['complaint_rate'],
    }]


def alert_description(row):
    month = row['month'][5:] + '/' + row['month'][:4]
    return (f"Tháng {month} có {row['count_NEG']}/{row['review_count']} nhận xét tiêu cực "
            f"về {row['aspect_name'].lower()}, tương ứng {pct(row['complaint_rate'])}. "
            f"Tỷ lệ này tăng {float(row['increase_pp']):.1f} điểm phần trăm so với "
            f"trung bình {row['baseline_months']} tháng trước ({pct(row['baseline_rate'])}).")


def empty_message(coverage):
    statuses = {row['status'] for row in coverage}
    if 'eligible' in statuses:
        return 'Không có cảnh báo xu hướng trong phạm vi đang xem.'
    if 'awaiting_predictions' in statuses:
        return 'Chưa có đủ kết quả phân tích nhận xét để đánh giá xu hướng.'
    if statuses == {'open_period'}:
        return 'Tháng đang xem chưa kết thúc.'
    if statuses:
        return 'Chưa đủ dữ liệu theo tháng để đánh giá xu hướng trong phạm vi đang xem.'
    return 'Không có dữ liệu theo tháng trong phạm vi đang xem.'


def render_card(row, master=None, evidence=None):
    """Hiển thị một kết quả của một khía cạnh và cửa sổ so sánh."""
    e = lambda v: escape(str(v))
    window = int(row['baseline_months'])
    months = comparison_months(row)
    table = ''.join(f"<tr><td>{e(b['month'])}</td><td>{e(b['negative_count'])}/{e(b['review_count'])}</td><td>{pct(b['complaint_rate'])}</td></tr>" for b in months)
    reviews = []
    for rid in json.loads(row['negative_review_ids_json']):
        review = (master or {}).get(rid)
        if not review:
            continue
        ev = (evidence or {}).get((rid, row['aspect']))
        title = 'Bằng chứng tay đã xác nhận' if ev and ev.get('evidence_source') == 'manual' else 'Bằng chứng do mô hình trích'
        excerpt = (f"<p><b>{title}:</b> {e(ev['evidence_text'])}</p>" if ev else '')
        reviews.append(f"<details><summary>{e(rid)} · {e(review['review_date_iso'][:10])}</summary><p class='original'>{e(review.get('review_text') or review['review_text_nlp'])}</p>{excerpt}</details>")
    evidence_html = (f"<details><summary>Đọc {len(reviews)} nhận xét liên quan</summary>{''.join(reviews)}</details>"
                     if reviews else '')
    return f"""<details class="event {e(row['level'])}">
<summary>{e(row['month'])} · {e(row['level_title'])} · So với {window} tháng trước</summary>
<p>{e(alert_description(row))}</p>
{render_chart(months, reference_rate=row['baseline_rate'])}
<div class="scroll"><table><thead><tr><th>Tháng</th><th>Tiêu cực / tổng nhận xét</th><th>Tỷ lệ tiêu cực</th></tr></thead><tbody>{table}</tbody></table></div>
{evidence_html}</details>"""


def group_alerts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['restaurant_id']].append(row)
    return list(groups.values())


def render_restaurant(rows, master=None, evidence=None):
    aspects = defaultdict(list)
    for row in rows:
        aspects[row['aspect']].append(row)
    sections = []
    for _, events in sorted(aspects.items()):
        heading = escape(events[0]['aspect_name'])
        details = ''.join(render_card(row, master, evidence) for row in events)
        sections.append(f'<section><h3>{heading}</h3>{details}</section>')
    return f"<article class='card'><h2>{escape(rows[0]['restaurant_name'])}</h2>{''.join(sections)}</article>"


STYLE = """
.foody-tang2{background:#f2f5f9;color:#1a2e49;font:15px/1.6 system-ui,sans-serif;color-scheme:light;overflow-wrap:anywhere}
.foody-tang2 main{max-width:1200px;margin:24px auto;padding:0 20px}.foody-tang2 h1,.foody-tang2 h2,.foody-tang2 h3{color:#1a2e49}
.foody-tang2 .muted{color:#526780}
.foody-tang2 select{font:inherit;padding:8px;border:1px solid #bdcbdc;border-radius:6px;background:white;color:#1a2e49;max-width:100%}
.foody-tang2 .card{padding:18px 22px;background:white;border:1px solid #d6dfeb;border-radius:10px;margin:18px 0}
.foody-tang2 details{margin:12px 0}.foody-tang2 summary{cursor:pointer;font-weight:650}.foody-tang2 .event{padding:12px;border-left:4px solid #d28c18}.foody-tang2 .event.red{border-left-color:#ba3d32}
.foody-tang2 table{width:100%;border-collapse:collapse;background:white}.foody-tang2 th,.foody-tang2 td{padding:8px;border-bottom:1px solid #d6dfeb;text-align:left;color:#1a2e49}.foody-tang2 .scroll{overflow:auto}
.foody-tang2 .original{white-space:pre-wrap;background:#f4f7fb;padding:12px}.foody-tang2 .trend{width:100%;min-width:320px;background:white;fill:#526780}.foody-tang2 [hidden]{display:none!important}
"""


def render_dashboard(result):
    """Bản HTML chọn một nhà hàng, rồi hiện toàn bộ cảnh báo của nhà hàng đó."""
    alerts = {rows[0]['restaurant_id']: rows for rows in group_alerts(result['alerts'])}
    restaurants = {}
    coverage = defaultdict(list)
    for row in result['coverage']:
        rid = row['restaurant_id']
        restaurants[rid] = row['restaurant_name']
        coverage[rid].append(row)
    for rid, rows in alerts.items():
        restaurants.setdefault(rid, rows[0]['restaurant_name'])

    options = ['<option value="">Chọn nhà hàng</option>']
    cards = {}
    for rid, name in sorted(restaurants.items(), key=lambda item: (item[1].casefold(), item[0])):
        options.append(f'<option value="{escape(rid)}">{escape(name)} · {escape(rid)}</option>')
        if rid in alerts:
            cards[rid] = render_restaurant(alerts[rid], result.get('master'), result.get('evidence'))
        else:
            cards[rid] = f'<h2>{escape(name)}</h2><p>{escape(empty_message(coverage[rid]))}</p>'

    data = json.dumps(cards, ensure_ascii=False).replace('<', '\\u003c').replace('&', '\\u0026')
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cảnh báo tầng 2</title>
<style>{STYLE}</style></head><body class="foody-tang2"><main>
<h1>Cảnh báo tầng 2 — Xu hướng theo tháng</h1>
<label for="restaurant">Nhà hàng</label>
<select id="restaurant">{''.join(options)}</select>
<div id="restaurant-alerts" aria-live="polite"></div></main>
<script id="payload" type="application/json">{data}</script>
<script>
const cards = JSON.parse(document.getElementById('payload').textContent);
const selection = document.getElementById('restaurant');
const output = document.getElementById('restaurant-alerts');
function showRestaurant() {{
    output.innerHTML = Object.hasOwn(cards, selection.value) ? cards[selection.value] : '';
}}
selection.addEventListener('change', showRestaurant);
showRestaurant();
</script></body></html>"""


class BoDocTang2:
    def __init__(self, project_root=ROOT, config=None, profile=None, as_of=None, *, system_inputs=None):
        cfg = load_settings(config, project_root)
        self.policy = policy_for(cfg, profile)
        inputs = system_inputs if system_inputs is not None else load_system_inputs(
            cfg['_root'], view='operational', config=cfg['_config'])
        cfg['_system_snapshot'] = inputs['paths']['directory']
        root = output_root(cfg)
        self.directory = current_directory(root)
        cutoff = as_of_time(as_of or datetime.now(ZoneInfo(self.policy['timezone'])).date().isoformat(), self.policy['timezone'])
        self.manifest = verify_run(self.directory, project_root=cfg['_root'])
        expected = signature(cfg, inputs, self.policy, cutoff)
        stored = dict(self.manifest['signature'])
        # Đổi giao diện vẫn đọc được các bảng cũ đã kiểm tra mã băm.
        # Bộ xây dựng vẫn dùng đầy đủ chữ ký để tạo mới HTML khi chạy lại.
        expected.pop('view_sha256')
        stored.pop('view_sha256', None)
        require(stored == expected, 'Tầng 2 chưa khớp dữ liệu/cấu hình hiện tại. Chạy lại ô tạo kết quả.')
        self.report_directory = report_directory_from_manifest(self.manifest, cfg['_root'])
        self.html_path = self.report_directory / 'demo_tang_2.html'
        self.master = inputs['master']
        self.evidence = {(r['review_id'], r['aspect']): r for r in inputs['aspects'] if r.get('evidence_text') and
                         ((r.get('label_source') == 'manual' and truth(r.get('label_verified')) and r.get('evidence_source') == 'manual')
                          or (r.get('label_source') != 'manual' and r.get('evidence_source') == 'model'
                              and r.get('label') == r.get('model_label') and r.get('evidence_model_version')))}

    def thong_ke(self):
        return dict(self.manifest['summary'])

    def _rows(self, name, restaurant_id=None, month=None, aspect=None, level=None, baseline_months=None):
        filters = [('restaurant_id',restaurant_id),('month',month),('aspect',aspect),('level',level),('baseline_months',baseline_months)]
        with (self.directory/name).open(encoding='utf-8-sig',newline='') as f:
            return [r for r in csv.DictReader(f) if all(v is None or v == '' or r.get(k) == str(v) for k,v in filters)]

    def danh_sach(self, restaurant_id=None, month=None, aspect=None, level=None, baseline_months=None):
        return self._rows('trend_alerts.csv', restaurant_id, month, aspect, level, baseline_months)

    def chi_so(self, restaurant_id=None, month=None, aspect=None, baseline_months=None):
        return self._rows('trend_metrics.csv', restaurant_id, month, aspect, baseline_months=baseline_months)

    def nha_hang_co_canh_bao(self, **filters):
        return [{'restaurant_id': rows[0]['restaurant_id'], 'restaurant_name': rows[0]['restaurant_name'], 'alerts': rows} for rows in group_alerts(self.danh_sach(**filters))]

    def hien_thi(self, restaurant_id=None, month=None, aspect=None, level=None, limit=10, baseline_months=None):
        """Xem một nhà hàng. Giữ đối số limit để các notebook cũ vẫn gọi được."""
        rows = self.danh_sach(restaurant_id, month, aspect, level, baseline_months)
        if not restaurant_id and rows:
            restaurant_id = rows[0]['restaurant_id']
            rows = [row for row in rows if row['restaurant_id'] == restaurant_id]
        html = f"<style>{STYLE}</style><div class='foody-tang2'><main>"
        if rows:
            html += render_restaurant(rows, self.master, self.evidence)
        elif restaurant_id:
            coverage = self.do_phu(restaurant_id, month, baseline_months)
            name = next((r['restaurant_name'] for r in self.master.values()
                         if r['restaurant_id'] == restaurant_id), restaurant_id)
            html += f'<h2>{escape(name)}</h2><p>{escape(empty_message(coverage))}</p>'
        else:
            html += '<p>Nhập mã nhà hàng để xem xu hướng.</p>'
        html += '</main></div>'
        from IPython.display import HTML, display
        display(HTML(html))
        return rows

    def do_phu(self, restaurant_id=None, month=None, baseline_months=None):
        return self._rows('trend_coverage.csv',restaurant_id,month,baseline_months=baseline_months)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--profile',default=None)
    parser.add_argument('--as-of',default=None)
    args=parser.parse_args(argv)
    reader=BoDocTang2(args.project_root,profile=args.profile,as_of=args.as_of)
    print(json.dumps(reader.thong_ke(),ensure_ascii=False,indent=2))
    print('Mở trong trình duyệt:',reader.html_path)

if __name__=='__main__':main()
