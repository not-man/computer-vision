"""Запасной офлайн-отчёт: один HTML без сервера. Plotly встроен, ролики лежат рядом и открываются по относительным путям."""
import base64
import json
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from mct.charts import camera_label, plan_figure, robustness_curves, timeline_figure
from mct.summary import (
    HONESTY_NOTE, KNOWN_ERRORS, camera_numbers, close_text, duration_s, hint_texts, link_rows, robustness_tiles, rule_rows, segment_rows, track_rows,
)
from mct.video import clock_text

CROP_HEIGHT = 120

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Сквозной трекинг</title>
<style>
:root{color-scheme:dark;--bg:#1a1a19;--panel:#262624;--text:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--selected:#27405e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:30px;margin:0 0 8px}h2{font-size:22px;margin:40px 0 12px}h3{font-size:18px;margin:24px 0 8px}
p{margin:0 0 12px}.muted{color:var(--muted)}
nav{display:flex;flex-wrap:wrap;gap:6px 16px;margin:0 0 24px}a{color:var(--text)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:0 0 16px}
.tile{background:var(--panel);border-radius:8px;padding:12px 16px;color:var(--muted);font-size:14px}
.tile b{display:block;color:var(--text);font-size:30px;font-weight:600;font-variant-numeric:tabular-nums}
video{width:100%;border-radius:8px;background:#000;display:block}
img{max-width:100%;display:block}
.scroll{overflow:auto;border:1px solid var(--line);border-radius:8px}.tall{max-height:440px}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
thead th{position:sticky;top:0;background:var(--panel);font-weight:600}
tr[tabindex]{cursor:pointer}tr[tabindex]:hover{background:var(--panel)}tr[tabindex]:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
tr.selected,tr.selected:hover{background:var(--selected)}
.columns{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;margin-top:16px}
.overview{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:16px}
.stack{display:grid;gap:16px;align-content:start}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
@media(max-width:800px){.columns,.overview{grid-template-columns:minmax(0,1fr)}}
.crops{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.crops figure{margin:0}.crops img{height:120px;border-radius:4px}
figcaption{color:var(--muted);font-size:12px;margin-top:2px}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 8px}
button,select{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 12px}
button{cursor:pointer}button:hover{border-color:var(--muted)}
</style></head>
<body><main>
<h1>Сквозной трекинг людей по камерам офиса</h1>
<p class="muted">Запасной отчёт без сервера. Держите файл в одной папке с роликами: он открывает их по относительным путям.</p>
<nav><a href="#showcase">Витрина</a><a href="#recording">Вся запись</a><a href="#tracks">Треки</a><a href="#returns">Возвраты</a><a href="#honesty">Как проверяли</a></nav>
<div class="tiles" id="summary"></div>

<h2 id="showcase">Витрина</h2>
<video controls preload="metadata" src="showcase.mp4"></video>
<p class="muted">Фрагменты полной записи. Вся запись ниже.</p>

<h2 id="recording">Вся запись</h2>
<div class="controls"><button id="previous">Назад</button><select id="minute" aria-label="Минута записи"></select><button id="next">Вперёд</button></div>
<video id="recording-video" controls preload="metadata"></video>
<p class="muted">Секунды в углах ячеек могут различаться: это часы камер. Кадры выровнены по общему времени.</p>
<div class="scroll tall" id="events"></div>

<h2 id="tracks">Треки</h2>
<p class="muted">Выберите строку, чтобы открыть досье трека.</p>
<div class="overview"><div class="scroll tall" id="track-table"></div><div id="timeline"></div></div>
<div id="dossier" hidden>
  <h3 id="dossier-title"></h3>
  <div class="tiles" id="dossier-tiles"></div>
  <div id="dossier-notes"></div>
  <div class="crops" id="dossier-crops"></div>
  <div class="columns">
    <div class="stack">
      <div class="controls" id="dossier-controls"><select id="dossier-minute" aria-label="Минута записи"></select></div>
      <video id="dossier-video" controls preload="metadata"></video>
      <div class="scroll" id="dossier-links"></div>
    </div>
    <div class="stack"><div class="scroll" id="dossier-segments"></div><div id="dossier-plan"></div></div>
  </div>
</div>

<h2 id="returns">Возвраты</h2>
<p>Слева ушедший трек, справа новый. Подсказка никогда не меняет ID и подсчёт: по ТЗ вернувшийся получает новый номер.</p>
<img id="hints" alt="Снимки ушедших и новых треков, которые похожи друг на друга">

<h2 id="honesty">Как проверяли</h2>
<p id="honesty-note"></p>
<div class="tiles" id="honesty-tiles"></div>
<div class="pair"><div id="curve-0"></div><div id="curve-1"></div></div>
<h3>Что будет, если отключить правило</h3>
<div class="scroll" id="rules"></div>
<h3>Известные ошибки</h3>
<div id="errors"></div>
</main>
<script>/*PLOTLY*/</script>
<script>
const DATA = /*DATA*/;
const CONFIG = {displayModeBar: false, responsive: true};
const byId = id => document.getElementById(id);

function makeTable(rows, onSelect) {
  const table = document.createElement('table');
  if (!rows.length) return table;
  const head = table.createTHead().insertRow();
  for (const name of Object.keys(rows[0])) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = name;
    head.appendChild(cell);
  }
  const body = table.createTBody();
  rows.forEach(row => {
    const line = body.insertRow();
    for (const value of Object.values(row)) line.insertCell().textContent = value === null ? '' : value;
    if (onSelect) {
      line.tabIndex = 0;
      line.addEventListener('click', () => onSelect(row, line));
      line.addEventListener('keydown', event => { if (event.key === 'Enter') onSelect(row, line); });
    }
  });
  return table;
}

function tiles(target, items) {
  target.replaceChildren(...items.map(([label, value]) => {
    const tile = document.createElement('div');
    tile.className = 'tile';
    const number = document.createElement('b');
    number.textContent = value;
    tile.append(label, number);
    return tile;
  }));
}

function plot(id, figure) { Plotly.react(byId(id), figure.data, figure.layout, CONFIG); }

function fillOptions(select, labels) {
  select.replaceChildren(...labels.map((label, index) => new Option(label, index)));
}

function showMinute(index) {
  const chunks = DATA.chunks;
  const chunk = chunks[Math.max(0, Math.min(chunks.length - 1, index))];
  byId('minute').value = chunks.indexOf(chunk);
  byId('recording-video').src = chunk.file;
  byId('events').replaceChildren(makeTable(chunk.events));
}

function showTrack(row, line) {
  document.querySelectorAll('#track-table tr.selected').forEach(other => other.classList.remove('selected'));
  line.classList.add('selected');
  const track = DATA.tracks[row['Трек']];
  byId('dossier').hidden = false;
  byId('dossier-title').textContent = 'Трек #' + row['Трек'];
  tiles(byId('dossier-tiles'), track.tiles);
  byId('dossier-notes').replaceChildren(...track.notes.map(text => { const note = document.createElement('p'); note.textContent = text; return note; }));
  byId('dossier-crops').replaceChildren(...track.crops.map(crop => {
    const figure = document.createElement('figure');
    const image = document.createElement('img');
    image.src = crop.uri;
    image.alt = 'Снимок трека: ' + crop.label;
    const caption = document.createElement('figcaption');
    caption.textContent = crop.label;
    figure.append(image, caption);
    return figure;
  }));
  fillOptions(byId('dossier-minute'), track.chunks.map(chunk => chunk.label));
  byId('dossier-controls').hidden = track.chunks.length < 2;
  byId('dossier-minute').onchange = event => openDossierVideo(track, Number(event.target.value));
  openDossierVideo(track, 0);
  byId('dossier-links').replaceChildren(track.links.length ? makeTable(track.links) : 'Трек видела одна камера, связей между камерами нет.');
  byId('dossier-segments').replaceChildren(makeTable(track.segments));
  const figure = JSON.parse(JSON.stringify(track.plan));  // Plotly меняет layout на месте, а досье открывают повторно
  figure.layout.images[0].source = DATA.plan_uri;
  figure.layout.width = byId('dossier-plan').clientWidth;  // без явной ширины масштаб плана считается по пустому контейнеру
  plot('dossier-plan', figure);
  Plotly.relayout(byId('timeline'), {shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: track.row, y1: track.row, line: {color: '#f0efec', width: 1}}]});
}

function openDossierVideo(track, index) {
  const chunk = track.chunks[index];
  byId('dossier-video').src = chunk.file + '#t=' + chunk.offset;
}

tiles(byId('summary'), DATA.summary);
fillOptions(byId('minute'), DATA.chunks.map(chunk => chunk.label));
byId('minute').addEventListener('change', event => showMinute(Number(event.target.value)));
byId('previous').addEventListener('click', () => showMinute(Number(byId('minute').value) - 1));
byId('next').addEventListener('click', () => showMinute(Number(byId('minute').value) + 1));
showMinute(0);
byId('track-table').replaceChildren(makeTable(DATA.rows, showTrack));
plot('timeline', DATA.timeline);
byId('hints').src = DATA.hints_uri;
byId('honesty-note').textContent = DATA.honesty_note;
tiles(byId('honesty-tiles'), DATA.honesty_tiles);
DATA.curves.forEach((figure, index) => plot('curve-' + index, figure));
byId('rules').replaceChildren(makeTable(DATA.rules));
byId('errors').replaceChildren(...DATA.errors.map(([title, text]) => {
  const item = document.createElement('p');
  const name = document.createElement('b');
  name.textContent = title + '. ';
  item.append(name, text);
  return item;
}));
</script>
</body></html>
"""


def _data_uri(payload: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def _figure(figure: go.Figure) -> dict:
    return json.loads(figure.to_json())


def _crops(tracks_dir: Path) -> dict[str, dict]:
    index_path = tracks_dir / "crops" / "index.json"
    if not index_path.exists():
        return {}
    crops = {}
    for tracklet, item in json.loads(index_path.read_text()).items():
        image = cv2.imread(str(tracks_dir / "crops" / item["file"]))
        if image is None:
            continue
        small = cv2.resize(image, (max(1, round(image.shape[1] * CROP_HEIGHT / image.shape[0])), CROP_HEIGHT), interpolation=cv2.INTER_AREA)
        crops[tracklet] = {
            "uri": _data_uri(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()),
            "label": f"{camera_label(item['camera'])} {clock_text(item['t_s'])[3:8]}",
            "t_s": item["t_s"],
        }
    return crops


def _track_chunks(track: dict, chunks: list[dict]) -> list[dict]:
    start, end = track["t_start_ms"] / 1000, track["t_end_ms"] / 1000
    return [{"label": chunk["start"][:8], "file": chunk["file"], "offset": round(max(start - chunk["t_start_s"] - 1, 0), 1)}
            for chunk in chunks if chunk["t_start_s"] < end and start < chunk["t_end_s"]]


def _track_entry(track: dict, row: int, crops: dict[str, dict], chunks: list[dict], plan_image: Path, configs: Path) -> dict:
    plan = _figure(plan_figure(track, plan_image, configs))
    plan["layout"]["images"][0]["source"] = ""  # одна картинка плана на весь отчёт, она подставляется при показе
    own_crops = sorted((crops[str(tracklet)] for tracklet in track["tracklets"] if str(tracklet) in crops), key=lambda crop: crop["t_s"])
    return {
        "row": row,
        "tiles": [("Начало", track["start"][:8]), ("Конец", track["end"][:8]), ("Длительность", f"{duration_s(track):.0f} с"), ("Камеры", camera_numbers(track))],
        "notes": [f"Закрыт: {close_text(track)}.", *hint_texts(track)],
        "crops": [{"uri": crop["uri"], "label": crop["label"]} for crop in own_crops],
        "chunks": _track_chunks(track, chunks),
        "links": link_rows(track),
        "segments": segment_rows(track),
        "plan": plan,
    }


def build_report(tracks_doc: dict, manifest: dict, robustness: pd.DataFrame, tracks_dir: Path, plan_image: Path, configs: Path, out: Path) -> None:
    tracks = tracks_doc["tracks"]
    chunks = manifest["chunks"]
    crops = _crops(tracks_dir)
    hints = tracks_dir / "gallery" / "hints.jpg"
    length_min = (manifest["clock_end_s"] - manifest["clock_start_s"]) / 60
    data = {
        "summary": [("Треков по ТЗ", str(tracks_doc["total_tracks"])),
                    ("Камер", str(len({camera for track in tracks for camera in track["cameras"]}))),
                    ("Длина записи", f"{length_min:.0f} мин"),
                    ("Подсказок о возврате", str(sum(len(track["possibly_same_person_as"]) for track in tracks)))],
        "chunks": [{"label": chunk["start"][:8], "file": chunk["file"],
                    "events": [{"Время": clock_text(event["t_s"])[:8], "Событие": event["text"]} for event in manifest["events"]
                               if chunk["t_start_s"] <= event["t_s"] < chunk["t_end_s"]]} for chunk in chunks],
        "rows": track_rows(tracks),
        "tracks": {str(track["track_id"]): _track_entry(track, row, crops, chunks, plan_image, configs) for row, track in enumerate(tracks)},
        "timeline": _figure(timeline_figure(tracks)),
        "plan_uri": _data_uri(plan_image.read_bytes()),
        "hints_uri": _data_uri(hints.read_bytes()) if hints.exists() else "",
        "honesty_note": HONESTY_NOTE,
        "honesty_tiles": robustness_tiles(robustness),
        "curves": [_figure(figure) for figure in robustness_curves(robustness)],
        "rules": rule_rows(robustness),
        "errors": [(error.title, error.text) for error in KNOWN_ERRORS],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out.write_text(PAGE.replace("/*PLOTLY*/", get_plotlyjs().replace("</script", "<\\/script")).replace("/*DATA*/", payload))
