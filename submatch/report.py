from __future__ import annotations
import csv
import dataclasses
import io
import json
import sys
from pathlib import Path
from typing import Any

from submatch.output import BatchPairResult


class _PathEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _write(path: str, content: str) -> None:
    try:
        Path(path).write_text(content, encoding="utf-8")
    except OSError as exc:
        print(f"Error: could not write {path}: {exc}", file=sys.stderr)
        sys.exit(2)


def write_json(results: list[BatchPairResult], path: str) -> None:
    items = []
    for p in results:
        d = dataclasses.asdict(p.result) if p.result is not None else {}
        d["video"] = str(p.video)
        d["subtitle"] = str(p.subtitle)
        if p.error is not None:
            d["error"] = p.error
        items.append(d)
    _write(path, json.dumps(items, cls=_PathEncoder, indent=2))


def write_csv(results: list[BatchPairResult], path: str) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "video", "subtitle", "state", "score", "threshold",
        "audio_lang", "subtitle_lang", "drift_detected", "cross_language", "error",
    ])
    for p in results:
        if p.result is None:
            writer.writerow([
                str(p.video), str(p.subtitle), "ERROR", "", "", "", "", "", "", p.error,
            ])
        else:
            r = p.result
            drift = r.sync.drift_detected if r.sync else False
            writer.writerow([
                str(p.video),
                str(p.subtitle),
                r.state.value,
                f"{r.confidence:.2f}",
                f"{r.threshold:.2f}",
                r.language.audio or "",
                r.subtitle_language or "",
                str(drift).lower(),
                str(r.cross_language).lower(),
                "",
            ])
    _write(path, buf.getvalue())


def write_html(results: list[BatchPairResult], path: str) -> None:
    import html as _esc
    from collections import Counter
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state_counts: Counter = Counter()

    rows_html: list[str] = []
    for p in results:
        if p.result is None:
            state_counts["ERROR"] += 1
            row_class = "row-error"
            state_label = "ERROR"
            score_str = threshold_str = audio_lang = sub_lang = ""
            error_str = _esc.escape(p.error or "")
        else:
            r = p.result
            state_counts[r.state.value] += 1
            row_class = {
                "PASS": "row-pass", "DRIFT": "row-drift",
                "FAIL": "row-fail",  "UNSURE": "row-unsure",
            }.get(r.state.value, "")
            state_label = r.state.value
            score_str = f"{r.confidence:.2f}"
            threshold_str = f"{r.threshold:.2f}"
            audio_lang = _esc.escape(r.language.audio or "")
            sub_lang = _esc.escape(r.subtitle_language or "")
            error_str = ""

        rows_html.append(
            f'<tr class="{row_class}">'
            f'<td>{_esc.escape(str(p.video))}</td>'
            f'<td>{_esc.escape(str(p.subtitle))}</td>'
            f'<td>{state_label}</td>'
            f'<td>{score_str}</td>'
            f'<td>{audio_lang}</td>'
            f'<td>{sub_lang}</td>'
            f'<td>{error_str}</td>'
            f'</tr>'
        )

    badge_colors = {
        "PASS": "#4caf50", "DRIFT": "#ff9800", "FAIL": "#f44336",
        "UNSURE": "#ff9800", "ERROR": "#e91e63",
    }
    badges_html = " ".join(
        f'<span style="background:{badge_colors.get(s,"#999")};color:white;'
        f'padding:2px 8px;border-radius:12px;margin:2px">{count} {s}</span>'
        for s, count in state_counts.most_common()
    )

    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>submatch report</title>
<style>
body{{font-family:sans-serif;padding:16px}}
h1{{font-size:1.4em}}
.summary{{margin:8px 0 16px}}
#filter{{width:300px;padding:6px;margin-bottom:12px}}
table{{border-collapse:collapse;width:100%;font-size:.9em}}
th{{background:#333;color:#fff;padding:8px;cursor:pointer;user-select:none}}
th:hover{{background:#555}}
td{{padding:6px 8px;border-bottom:1px solid #ddd}}
.row-pass td{{background:#e8f5e9}}
.row-drift td{{background:#fff8e1}}
.row-fail td{{background:#ffebee}}
.row-unsure td{{background:#fff8e1}}
.row-error td{{background:#fce4ec}}
tr:hover td{{filter:brightness(.95)}}
th.asc::after{{content:" ▲"}}
th.desc::after{{content:" ▼"}}
</style>
</head>
<body>
<h1>submatch report</h1>
<div class="summary"><span style="color:#666">{timestamp}</span> &nbsp;{badges_html}</div>
<input id="filter" type="text" placeholder="Filter…" oninput="filterTable()">
<table id="t">
<thead><tr>
<th onclick="sortTable(0)">Video</th>
<th onclick="sortTable(1)">Subtitle</th>
<th onclick="sortTable(2)">State</th>
<th onclick="sortTable(3)">Score</th>
<th onclick="sortTable(4)">Audio Lang</th>
<th onclick="sortTable(5)">Sub Lang</th>
<th onclick="sortTable(6)">Error</th>
</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
<script>
let _col=-1,_asc=true;
function sortTable(c){{
  const tb=document.querySelector('#t tbody');
  const rows=Array.from(tb.rows);
  if(_col===c){{_asc=!_asc;}}else{{_col=c;_asc=true;}}
  rows.sort((a,b)=>{{
    const av=a.cells[c].textContent,bv=b.cells[c].textContent;
    const n=parseFloat(av)-parseFloat(bv);
    if(!isNaN(n))return _asc?n:-n;
    return _asc?av.localeCompare(bv):bv.localeCompare(av);
  }});
  rows.forEach(r=>tb.appendChild(r));
  document.querySelectorAll('th').forEach((th,i)=>{{
    th.className=i===c?(_asc?'asc':'desc'):'';
  }});
}}
function filterTable(){{
  const q=document.getElementById('filter').value.toLowerCase();
  document.querySelectorAll('#t tbody tr').forEach(r=>{{
    r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';
  }});
}}
</script>
</body>
</html>"""
    _write(path, content)
