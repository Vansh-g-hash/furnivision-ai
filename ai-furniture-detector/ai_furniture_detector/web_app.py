from __future__ import annotations

import base64
import csv
import io
import json
import socket
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

try:
    from .detector import (
        DEFAULT_CONFIDENCE,
        DEFAULT_IOU,
        build_amazon_link,
        category_for_item,
        default_model_path,
        display_name,
        load_model,
        run_detection,
    )
except ImportError:  # Allows: python web_app.py
    from detector import (  # type: ignore
        DEFAULT_CONFIDENCE,
        DEFAULT_IOU,
        build_amazon_link,
        category_for_item,
        default_model_path,
        display_name,
        load_model,
        run_detection,
    )

APP_TITLE = "FurniVision AI — Furniture Detection Studio"
MAX_UPLOAD_MB = 15
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

app = FastAPI(title=APP_TITLE, version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_industry_headers(request, call_next):
    """Add lightweight production-friendly headers for browser safety and tracing."""
    request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@lru_cache(maxsize=3)
def cached_model(model_path: str):
    return load_model(Path(model_path).expanduser())


def decode_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise ValueError("Uploaded image is empty")
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image. Use JPG, PNG, WEBP, or JPEG.")
    return image


def encode_jpeg_data_url(image_bgr: np.ndarray, quality: int = 94) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode image")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def safe_bbox(raw_bbox: Any) -> list[int]:
    try:
        values = list(raw_bbox or [0, 0, 0, 0])[:4]
        while len(values) < 4:
            values.append(0)
        return [int(round(float(v))) for v in values]
    except Exception:
        return [0, 0, 0, 0]


def clean_item_dict(item: Any, index: int) -> dict[str, Any]:
    data = item.to_dict() if hasattr(item, "to_dict") else dict(item)
    raw_name = str(data.get("name", "unknown")).strip().lower()
    pretty_name = display_name(raw_name)
    category = str(data.get("category") or category_for_item(raw_name) or "Other")
    link = str(data.get("link") or build_amazon_link(raw_name))
    confidence = float(data.get("confidence", 0) or 0)
    count = int(data.get("count", 1) or 1)
    bbox = safe_bbox(data.get("bbox") or data.get("coords") or [0, 0, 0, 0])
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return {
        "id": f"det-{index}",
        "name": raw_name,
        "display_name": pretty_name,
        "category": category,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "count": count,
        "bbox": bbox,
        "area": area,
        "link": link,
    }


def build_summary(items: list[dict[str, Any]], all_items: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    link_count = 0
    for item in items:
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + int(item.get("count", 1))
        if item.get("link"):
            link_count += 1
    top_category = max(category_counts, key=category_counts.get) if category_counts else "None"
    avg_conf = round(sum(i.get("confidence", 0) for i in all_items) / len(all_items) * 100, 1) if all_items else 0
    return {
        "unique_items": len(items),
        "total_detections": len(all_items),
        "category_counts": category_counts,
        "top_category": top_category,
        "avg_confidence": avg_conf,
        "shopping_links": link_count,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": APP_TITLE}


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots() -> str:
    return "User-agent: *\nDisallow:\n"


@app.post("/api/detect")
async def detect_image(
    image: UploadFile = File(...),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    iou: float = Form(DEFAULT_IOU),
    model_path: str = Form(str(default_model_path())),
) -> JSONResponse:
    try:
        if confidence < 0 or confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if iou < 0 or iou > 1:
            raise ValueError("IoU must be between 0 and 1")

        if image.content_type and image.content_type.lower() not in ALLOWED_IMAGE_TYPES:
            raise ValueError("Unsupported image type. Use JPG, PNG, JPEG, or WEBP.")

        started_at = time.perf_counter()
        file_bytes = await image.read()
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Image is too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.")

        image_bgr = decode_upload_to_bgr(file_bytes)
        model = cached_model(model_path)
        result = run_detection(model, image_bgr, confidence=confidence, iou=iou)
        process_ms = round((time.perf_counter() - started_at) * 1000, 2)

        items = [clean_item_dict(item, idx) for idx, item in enumerate(result.unique_items)]
        all_items = [clean_item_dict(item, idx) for idx, item in enumerate(result.all_items)]
        items.sort(key=lambda x: (x.get("category", ""), -float(x.get("confidence", 0)), x.get("display_name", "")))

        return JSONResponse(
            {
                "ok": True,
                "filename": image.filename or "uploaded_image",
                "original_image": encode_jpeg_data_url(image_bgr),
                "image": encode_jpeg_data_url(result.annotated_image),
                "image_width": int(result.annotated_image.shape[1]),
                "image_height": int(result.annotated_image.shape[0]),
                "unique_count": len(items),
                "all_count": len(all_items),
                "items": items,
                "all_items": all_items,
                "summary": build_summary(items, all_items),
                "process_ms": process_ms,
                "model": Path(model_path).name,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model file not found: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc


HTML_PAGE = r'''
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FurniVision AI</title>
<style>
:root{
  --bg:#070914;--bg2:#0b1020;--panel:rgba(15,23,42,.82);--panel2:rgba(2,6,23,.88);
  --card:rgba(255,255,255,.07);--card2:rgba(255,255,255,.105);--line:rgba(148,163,184,.22);
  --text:#f8fafc;--muted:#94a3b8;--soft:#cbd5e1;--brand:#38bdf8;--brand2:#a78bfa;
  --good:#22c55e;--warn:#f59e0b;--danger:#fb7185;--shadow:0 30px 80px rgba(0,0,0,.45);
}
*{box-sizing:border-box} html,body{height:100%} body{margin:0;color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:radial-gradient(circle at 18% 5%,rgba(56,189,248,.18),transparent 26%),radial-gradient(circle at 88% 14%,rgba(167,139,250,.16),transparent 24%),linear-gradient(135deg,#030712,#0f172a 52%,#020617);overflow:hidden}
button,input,select{font:inherit}.app{height:100vh;display:grid;grid-template-columns:310px minmax(0,1fr) 390px;gap:14px;padding:14px}.glass{background:linear-gradient(180deg,var(--panel),rgba(15,23,42,.68));border:1px solid var(--line);box-shadow:var(--shadow);backdrop-filter:blur(18px);border-radius:26px;overflow:hidden}.left,.right{display:flex;flex-direction:column;min-height:0}.brand{padding:20px;border-bottom:1px solid var(--line)}.logo{display:flex;align-items:center;gap:12px}.mark{width:44px;height:44px;border-radius:15px;background:linear-gradient(135deg,var(--brand),var(--brand2));display:grid;place-items:center;font-size:22px;box-shadow:0 12px 30px rgba(56,189,248,.25)}h1{font-size:18px;margin:0;letter-spacing:.2px}.sub{font-size:12px;color:var(--muted);margin-top:3px}.upload{margin:16px;padding:18px;border:1.5px dashed rgba(148,163,184,.35);border-radius:22px;background:rgba(255,255,255,.045);text-align:center;cursor:pointer;transition:.18s}.upload:hover,.upload.drag{border-color:var(--brand);background:rgba(56,189,248,.08);transform:translateY(-1px)}.upload .big{font-weight:900;font-size:15px}.upload p{margin:8px 0 0;color:var(--muted);font-size:12px;line-height:1.45}.hidden{display:none!important}.section{padding:0 16px 16px}.label{display:flex;justify-content:space-between;align-items:center;font-weight:900;font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:#e2e8f0;margin:16px 0 10px}.control{background:rgba(2,6,23,.42);border:1px solid var(--line);border-radius:16px;padding:13px;margin-bottom:10px}.control-row{display:flex;align-items:center;justify-content:space-between;gap:10px}.control b{font-size:13px}.value{font-size:12px;color:var(--brand);font-weight:900}.range{width:100%;accent-color:var(--brand);margin-top:10px}.primary,.secondary,.ghost{border:0;border-radius:16px;padding:12px 14px;color:var(--text);font-weight:900;cursor:pointer;transition:.16s}.primary{width:100%;background:linear-gradient(135deg,var(--brand),var(--brand2));box-shadow:0 18px 38px rgba(56,189,248,.23)}.primary:hover{transform:translateY(-1px)}.primary:disabled{opacity:.52;cursor:not-allowed;transform:none}.secondary{background:rgba(255,255,255,.08);border:1px solid var(--line)}.ghost{background:transparent;border:1px solid var(--line);color:var(--soft)}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}.mini{font-size:12px;color:var(--muted);line-height:1.5}.center{display:grid;grid-template-rows:auto minmax(0,1fr) auto;min-width:0}.topbar{height:72px;display:flex;align-items:center;gap:10px;padding:12px 14px;border-bottom:1px solid var(--line)}.pill{display:inline-flex;align-items:center;gap:8px;padding:9px 12px;background:rgba(255,255,255,.075);border:1px solid var(--line);border-radius:999px;font-size:12px;font-weight:900;color:var(--soft)}.spacer{flex:1}.viewbtn.active{background:rgba(56,189,248,.16);border-color:rgba(56,189,248,.5);color:#e0f2fe}.canvas-wrap{min-height:0;display:grid;place-items:center;overflow:auto;padding:22px;background:radial-gradient(circle at 50% 20%,rgba(255,255,255,.04),transparent 24%)}.stage{position:relative;max-width:100%;max-height:100%;transform-origin:center;transition:transform .12s ease}.imgbox{position:relative;display:block}.main-img{display:block;max-width:min(100%,calc(100vw - 760px));max-height:calc(100vh - 160px);width:auto;height:auto;border-radius:18px;box-shadow:0 24px 70px rgba(0,0,0,.45);background:#111}.hotspot{position:absolute;border:2px solid rgba(56,189,248,.0);background:rgba(56,189,248,.0);border-radius:10px;transition:.13s;pointer-events:auto;cursor:pointer}.hotspot:hover,.hotspot.active{border-color:rgba(56,189,248,.95);background:rgba(56,189,248,.16);box-shadow:0 0 0 4px rgba(56,189,248,.12),0 0 26px rgba(56,189,248,.35)}.hotspot-label{position:absolute;z-index:120;display:inline-flex;align-items:center;gap:5px;white-space:nowrap;text-decoration:none;color:#f8fafc;background:#0b1b2f;border:1px solid rgba(56,189,248,.48);border-radius:999px;padding:5px 8px;font-size:10.5px;font-weight:1000;line-height:1;box-shadow:0 8px 18px rgba(0,0,0,.30),inset 0 -2px 0 rgba(56,189,248,.18);cursor:pointer;transition:.14s}.hotspot-label::after{content:'↗';color:#facc15;font-size:11px;font-weight:1000}.hotspot-label:hover,.hotspot-label.active{background:#12345a;border-color:rgba(250,204,21,.9);box-shadow:0 0 0 4px rgba(250,204,21,.14),0 12px 28px rgba(0,0,0,.45);transform:translateY(-1px)}.leader-line{position:absolute;height:2px;background:linear-gradient(90deg,rgba(56,189,248,.15),rgba(56,189,248,.9));transform-origin:0 50%;border-radius:99px;pointer-events:none;z-index:80}.leader-line.active{height:3px;background:linear-gradient(90deg,rgba(250,204,21,.2),rgba(250,204,21,.95));box-shadow:0 0 14px rgba(250,204,21,.5)}.empty{width:min(760px,90%);min-height:460px;display:grid;place-items:center;text-align:center;border:1px dashed rgba(148,163,184,.35);border-radius:28px;background:rgba(255,255,255,.04);padding:34px}.empty h2{margin:0 0 8px;font-size:32px}.empty p{margin:0;color:var(--muted);line-height:1.6}.footerbar{height:60px;border-top:1px solid var(--line);display:flex;align-items:center;gap:10px;padding:10px 14px;color:var(--muted);font-size:12px}.zoom{width:160px;accent-color:var(--brand)}.right-head{padding:18px;border-bottom:1px solid var(--line)}.search{width:100%;padding:13px 14px;border-radius:16px;border:1px solid var(--line);background:rgba(2,6,23,.44);color:var(--text);outline:none}.stats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}.stat{background:rgba(255,255,255,.065);border:1px solid var(--line);border-radius:18px;padding:13px}.stat span{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.stat b{font-size:22px}.chips{display:flex;gap:8px;overflow:auto;padding:12px 18px;border-bottom:1px solid var(--line)}.chip{flex:0 0 auto;border:1px solid var(--line);background:rgba(255,255,255,.06);color:var(--soft);border-radius:999px;padding:8px 10px;font-size:12px;font-weight:900;cursor:pointer}.chip.active{background:rgba(56,189,248,.18);border-color:rgba(56,189,248,.52);color:#e0f2fe}.items{padding:14px 14px 18px;overflow:auto;min-height:0}.card{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;margin-bottom:10px;padding:13px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.055);cursor:pointer;transition:.14s}.card:hover,.card.active{transform:translateY(-1px);background:rgba(56,189,248,.11);border-color:rgba(56,189,248,.42)}.card h3{margin:0;font-size:14px}.meta{margin-top:6px;display:flex;flex-wrap:wrap;gap:6px}.tag{font-size:11px;color:var(--soft);background:rgba(255,255,255,.08);border:1px solid rgba(148,163,184,.18);border-radius:999px;padding:5px 8px}.buy{align-self:start;text-decoration:none;color:#020617;background:#facc15;border-radius:12px;padding:8px 10px;font-size:12px;font-weight:1000}.toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%) translateY(20px);opacity:0;background:#0f172a;border:1px solid var(--line);box-shadow:var(--shadow);padding:12px 16px;border-radius:16px;transition:.2s;z-index:20}.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}.loader{position:fixed;inset:0;background:rgba(2,6,23,.72);backdrop-filter:blur(12px);display:none;place-items:center;z-index:30}.loader.show{display:grid}.load-card{width:min(420px,90vw);padding:26px;border-radius:24px;background:#0f172a;border:1px solid var(--line);text-align:center}.spinner{width:54px;height:54px;border:5px solid rgba(255,255,255,.12);border-top-color:var(--brand);border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 16px}@keyframes spin{to{transform:rotate(360deg)}}.compare{position:relative;overflow:hidden;border-radius:18px}.compare img{display:block}.compare .after{position:absolute;inset:0;clip-path:inset(0 0 0 50%)}.compare input{position:absolute;inset:auto 18px 18px 18px;width:calc(100% - 36px);accent-color:var(--brand)}.compare-line{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 14px rgba(255,255,255,.6)}@media(max-width:1100px){body{overflow:auto}.app{height:auto;min-height:100vh;grid-template-columns:1fr}.center{min-height:680px}.main-img{max-width:calc(100vw - 72px)}.left,.right{min-height:auto}.topbar{flex-wrap:wrap;height:auto}.footerbar{height:auto;flex-wrap:wrap}} 
</style>
</head>
<body>
<div class="app">
  <aside class="left glass">
    <div class="brand"><div class="logo"><div class="mark">◆</div><div><h1>FurniVision AI</h1><div class="sub">Market-ready furniture detection studio</div></div></div></div>
    <label id="dropZone" class="upload" for="fileInput"><input id="fileInput" class="hidden" type="file" accept="image/*"><div class="big">Drop room image here</div><p>or click to upload JPG / PNG / WEBP. Your detector stays exactly same.</p></label>
    <div class="section">
      <div class="label"><span>Detection Settings</span><span id="modelState">Ready</span></div>
      <div class="control"><div class="control-row"><b>Confidence</b><span class="value" id="confValue"></span></div><input id="confidence" class="range" type="range" min="0.05" max="0.90" step="0.01" value="0.12"></div>
      <div class="control"><div class="control-row"><b>IoU / Overlap</b><span class="value" id="iouValue"></span></div><input id="iou" class="range" type="range" min="0.20" max="0.90" step="0.01" value="0.50"></div>
      <button id="detectBtn" class="primary" disabled>Detect Furniture</button>
      <div class="grid2" style="margin-top:10px"><button id="resetBtn" class="secondary">Reset</button><button id="demoTips" class="secondary">Tips</button></div>
      <p class="mini">Tip: for cleaner results, use a bright wide-angle room photo. Lower confidence catches more objects; higher confidence reduces false tags.</p>
    </div>
  </aside>

  <main class="center glass">
    <div class="topbar">
      <span class="pill" id="filePill">No image selected</span>
      <span class="pill" id="sizePill">—</span><span class="pill" id="speedPill">—</span>
      <div class="spacer"></div>
      <button class="secondary viewbtn active" data-view="detected">Detected</button>
      <button class="secondary viewbtn" data-view="original">Original</button>
      <button class="secondary viewbtn" data-view="compare">Before / After</button>
    </div>
    <div class="canvas-wrap" id="canvasWrap">
      <div class="empty" id="empty"><div><h2>Upload. Detect. Shop.</h2><p>Clean dashboard, real before/after, clickable products, filters, exports, hover highlights, upload validation, and faster cached model serving.</p></div></div>
      <div class="stage hidden" id="stage"><div class="imgbox" id="imageBox"><img id="mainImage" class="main-img" alt="Detection output"></div></div>
    </div>
    <div class="footerbar">
      <button id="fitBtn" class="ghost">Fit</button><button id="zoomOut" class="ghost">−</button><input id="zoomRange" class="zoom" type="range" min="50" max="200" value="100"><button id="zoomIn" class="ghost">+</button><span id="zoomLabel">100%</span><div class="spacer"></div><button id="downloadImg" class="ghost">Download Image</button><button id="copyLinks" class="ghost">Copy Links</button><button id="exportJson" class="ghost">JSON</button><button id="exportCsv" class="ghost">CSV</button>
    </div>
  </main>

  <aside class="right glass">
    <div class="right-head"><input id="search" class="search" placeholder="Search detected item..."><div class="stats"><div class="stat"><span>Unique</span><b id="uniqueStat">0</b></div><div class="stat"><span>Total</span><b id="totalStat">0</b></div><div class="stat"><span>Avg Conf.</span><b id="confStat">0%</b></div><div class="stat"><span>Top Cat.</span><b id="catStat" style="font-size:15px">—</b></div></div></div>
    <div class="chips" id="chips"><button class="chip active" data-cat="All">All</button></div>
    <div class="items" id="items"><p class="mini">Detected objects will appear here. Hover a card to highlight the object; click Buy to open product search.</p></div>
  </aside>
</div>
<div class="loader" id="loader"><div class="load-card"><div class="spinner"></div><h2>Detecting furniture...</h2><p class="mini">Model is analyzing your room image and preparing clickable results.</p></div></div>
<div class="toast" id="toast"></div>
<script>
const $=s=>document.querySelector(s); const $$=s=>Array.from(document.querySelectorAll(s));
let selectedFile=null,lastData=null,currentView='detected',activeCat='All',zoom=100,activeId=null;
const fileInput=$('#fileInput'),dropZone=$('#dropZone'),detectBtn=$('#detectBtn'),loader=$('#loader'),stage=$('#stage'),empty=$('#empty'),imageBox=$('#imageBox'),mainImage=$('#mainImage');
function toast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2200)}
function fmtPct(v){return Math.round(v*100)+'%'}
function updateSliders(){ $('#confValue').textContent=fmtPct($('#confidence').value); $('#iouValue').textContent=fmtPct($('#iou').value)}
['confidence','iou'].forEach(id=>$('#'+id).addEventListener('input',updateSliders)); updateSliders();
function setFile(file){ if(!file) return; if(!file.type.startsWith('image/')) return toast('Please upload an image file'); selectedFile=file; detectBtn.disabled=false; $('#filePill').textContent=file.name; const url=URL.createObjectURL(file); mainImage.src=url; empty.classList.add('hidden'); stage.classList.remove('hidden'); clearHotspots(); $('#items').innerHTML='<p class="mini">Image loaded. Press Detect Furniture.</p>'; $('#sizePill').textContent='Preview'; }
fileInput.addEventListener('change',e=>setFile(e.target.files[0]));
['dragenter','dragover'].forEach(ev=>dropZone.addEventListener(ev,e=>{e.preventDefault();dropZone.classList.add('drag')}));
['dragleave','drop'].forEach(ev=>dropZone.addEventListener(ev,e=>{e.preventDefault();dropZone.classList.remove('drag')}));
dropZone.addEventListener('drop',e=>setFile(e.dataTransfer.files[0]));
detectBtn.addEventListener('click',detect);
async function detect(){ if(!selectedFile) return; loader.classList.add('show'); detectBtn.disabled=true; $('#modelState').textContent='Running'; try{ const fd=new FormData(); fd.append('image',selectedFile); fd.append('confidence',$('#confidence').value); fd.append('iou',$('#iou').value); const res=await fetch('/api/detect',{method:'POST',body:fd}); const data=await res.json(); if(!res.ok||!data.ok) throw new Error(data.detail||'Detection failed'); lastData=data; currentView='detected'; activeCat='All'; activeId=null; renderAll(); toast('Detection complete'); }catch(err){ toast(err.message||'Something went wrong'); }finally{ loader.classList.remove('show'); detectBtn.disabled=false; $('#modelState').textContent='Ready'; }}
function renderAll(){ if(!lastData) return; $('#filePill').textContent=lastData.filename; $('#sizePill').textContent=`${lastData.image_width} × ${lastData.image_height}`; $('#speedPill').textContent=lastData.process_ms?`${lastData.process_ms} ms • ${lastData.model||'model'}`:'—'; $('#uniqueStat').textContent=lastData.summary.unique_items; $('#totalStat').textContent=lastData.summary.total_detections; $('#confStat').textContent=lastData.summary.avg_confidence+'%'; $('#catStat').textContent=lastData.summary.top_category; renderImage(); renderChips(); renderItems(); }
function setView(v){currentView=v; $$('.viewbtn').forEach(b=>b.classList.toggle('active',b.dataset.view===v)); renderImage();}
$$('.viewbtn').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
function clearHotspots(){ imageBox.querySelectorAll('.hotspot,.hotspot-label,.leader-line,.compare-line,.compare-range,.compare-img').forEach(x=>x.remove()); imageBox.classList.remove('compare'); }
function renderImage(){ clearHotspots(); if(!lastData) return; empty.classList.add('hidden'); stage.classList.remove('hidden'); if(currentView==='original'){ mainImage.src=lastData.original_image; return; } if(currentView==='compare'){ imageBox.classList.add('compare'); mainImage.src=lastData.original_image; const after=document.createElement('img'); after.src=lastData.image; after.className='main-img after compare-img'; const line=document.createElement('div'); line.className='compare-line'; const range=document.createElement('input'); range.className='compare-range'; range.type='range'; range.min=0; range.max=100; range.value=50; range.oninput=()=>{after.style.clipPath=`inset(0 0 0 ${range.value}%)`; line.style.left=range.value+'%'}; imageBox.append(after,line,range); return; } mainImage.src=lastData.original_image; mainImage.onload=renderHotspots; if(mainImage.complete) renderHotspots(); }
function clamp(n,min,max){return Math.max(min,Math.min(max,n))}
function itemArea(item){ const [x1,y1,x2,y2]=item.bbox; return Math.max(1,x2-x1)*Math.max(1,y2-y1); }
function smartHotspotBbox(item){
  // IMPORTANT: this changes ONLY the web hover/click rectangle, not detector output.
  // Some detections have very large boxes (window blind, whiteboard, conference table, floor).
  // On hover those boxes covered too much of the room, so this returns a MEDIUM sample rectangle.
  const W=lastData.image_width, H=lastData.image_height;
  let [x1,y1,x2,y2]=item.bbox.map(Number);
  x1=clamp(x1,0,W-1); y1=clamp(y1,0,H-1); x2=clamp(x2,x1+1,W); y2=clamp(y2,y1+1,H);

  const name=String(item.name||'').toLowerCase();
  const display=String(item.display_name||'').toLowerCase();
  const cat=String(item.category||'').toLowerCase();
  const bw=x2-x1, bh=y2-y1, imgArea=W*H, area=bw*bh;
  const isBig=area/imgArea>0.035 || bw/W>0.34 || bh/H>0.34;

  const mediumPatchNames=[
    'window blind','blind','roller blind','vertical blind',
    'whiteboard','conference table','table','floor','carpet','rug','floor rug','floor mat',
    'glass partition','glass window','glass door','door','window','wall','ceiling'
  ];
  const needsMediumPatch=isBig && (
    mediumPatchNames.some(k=>name.includes(k) || display.includes(k)) ||
    cat.includes('structure') || cat.includes('flooring') || cat.includes('doors')
  );

  if(!needsMediumPatch) return [x1,y1,x2,y2];

  let cx=(x1+x2)/2, cy=(y1+y2)/2;
  let pw=Math.round(clamp(bw*0.42,90,330));
  let ph=Math.round(clamp(bh*0.38,65,230));

  if(name.includes('whiteboard') || display.includes('whiteboard')){
    // Medium patch on the center-right of the board; not the full board.
    pw=Math.round(clamp(bw*0.42,110,300));
    ph=Math.round(clamp(bh*0.42,80,230));
    cx=x1+bw*0.58; cy=y1+bh*0.52;
  }else if(name.includes('conference table') || display.includes('conference table') || name==='table'){
    // Medium patch around the table center, not the whole table surface.
    pw=Math.round(clamp(bw*0.42,130,360));
    ph=Math.round(clamp(bh*0.34,90,240));
    cx=x1+bw*0.50; cy=y1+bh*0.55;
  }else if(name.includes('window blind') || name.includes('blind') || display.includes('window blind')){
    pw=Math.round(clamp(bw*0.48,120,340));
    ph=Math.round(clamp(bh*0.36,80,230));
    cx=x1+bw*0.50; cy=y1+bh*0.42;
  }else if(name.includes('floor')||name.includes('carpet')||name.includes('rug')||display.includes('floor')){
    pw=Math.round(clamp(bw*0.34,120,330));
    ph=Math.round(clamp(bh*0.28,75,200));
    cx=x1+bw*0.62; cy=y1+bh*0.70;
  }else if(name.includes('ceiling')){
    pw=Math.round(clamp(bw*0.30,100,300));
    ph=Math.round(clamp(bh*0.26,65,170));
    cx=x1+bw*0.50; cy=y1+bh*0.28;
  }else if(name.includes('glass')||name.includes('door')||name.includes('window')||cat.includes('doors')){
    pw=Math.round(clamp(bw*0.36,100,310));
    ph=Math.round(clamp(bh*0.42,95,280));
    cx=x1+bw*0.48; cy=y1+bh*0.50;
  }else if(name.includes('wall')){
    pw=Math.round(clamp(bw*0.32,100,290));
    ph=Math.round(clamp(bh*0.32,80,220));
    cx=x1+bw*0.52; cy=y1+bh*0.48;
  }

  let nx1=Math.round(cx-pw/2), ny1=Math.round(cy-ph/2);
  let nx2=nx1+pw, ny2=ny1+ph;
  nx1=clamp(nx1,x1,x2-1); ny1=clamp(ny1,y1,y2-1);
  nx2=clamp(nx2,nx1+1,x2); ny2=clamp(ny2,ny1+1,y2);
  return [nx1,ny1,nx2,ny2];
}
function labelOrder(a,b){
  // Place important/small labels first so they do not get pushed away by big surfaces.
  const pa=labelPriority(a), pb=labelPriority(b);
  if(pa!==pb) return pa-pb;
  return itemArea(a)-itemArea(b);
}
function labelPriority(item){
  const d=String(item.display_name||'').toLowerCase();
  const n=String(item.name||'').toLowerCase();
  if(d.includes('sideboard')||d.includes('landline')||d.includes('whiteboard')||d.includes('conference table')) return 0;
  if(d.includes('conference chair')||d.includes('chair')) return 1;
  if(d.includes('television')||d.includes('screen')||d.includes('lamp')) return 2;
  if(n.includes('ceiling')||n.includes('floor')||n.includes('glass')||n.includes('window')||n.includes('partition')) return 4;
  return 3;
}
function renderHotspots(){
  if(!lastData||currentView!=='detected') return;
  clearHotspots();
  const sx=mainImage.clientWidth/lastData.image_width, sy=mainImage.clientHeight/lastData.image_height;
  const items=filteredItems().slice().sort(labelOrder);
  const placed=[];
  const boxW=mainImage.clientWidth, boxH=mainImage.clientHeight;

  function overlaps(a,b,pad=6){return !(a.x+a.w+pad<b.x||b.x+b.w+pad<a.x||a.y+a.h+pad<b.y||b.y+b.h+pad<a.y)}
  function overlapScore(r){let score=0; placed.forEach(p=>{ if(overlaps(r,p,4)){ const ix=Math.max(0,Math.min(r.x+r.w,p.x+p.w)-Math.max(r.x,p.x)); const iy=Math.max(0,Math.min(r.y+r.h,p.y+p.h)-Math.max(r.y,p.y)); score+=ix*iy+5000; }}); return score;}
  function clampRect(r){r.x=clamp(r.x,4,Math.max(4,boxW-r.w-4)); r.y=clamp(r.y,4,Math.max(4,boxH-r.h-4)); return r;}
  function bestLabelRect(cx,cy,x1,y1,x2,y2,text){
    const w=Math.ceil(clamp(text.length*6.2+26,58,210));
    const h=24;
    const cands=[];
    const offsets=[8,18,30,44,60,78,98,122];
    offsets.forEach(off=>{
      cands.push({x:cx-w/2,y:y1-h-off,w,h});
      cands.push({x:cx-w/2,y:y2+off,w,h});
      cands.push({x:x2+off,y:cy-h/2,w,h});
      cands.push({x:x1-w-off,y:cy-h/2,w,h});
    });
    // edge lanes help crowded conference rooms without hiding the object itself
    cands.push({x:cx-w/2,y:8,w,h},{x:cx-w/2,y:boxH-h-8,w,h},{x:8,y:cy-h/2,w,h},{x:boxW-w-8,y:cy-h/2,w,h});
    let best=null,bestScore=Infinity;
    cands.forEach(raw=>{
      const r=clampRect({...raw});
      const dist=Math.hypot((r.x+r.w/2)-cx,(r.y+r.h/2)-cy);
      const edgePenalty=(r.x<8||r.y<8||r.x+r.w>boxW-8||r.y+r.h>boxH-8)?60:0;
      const score=overlapScore(r)+dist+edgePenalty;
      if(score<bestScore){bestScore=score;best=r;}
    });
    placed.push(best);
    return best;
  }
  function addLeader(id,x1,y1,x2,y2,lx,ly,lw,lh){
    const cx=(x1+x2)/2, cy=(y1+y2)/2;
    const tx=lx+lw/2, ty=ly+lh/2;
    const dx=tx-cx, dy=ty-cy;
    const len=Math.max(10,Math.hypot(dx,dy));
    const line=document.createElement('div');
    line.className='leader-line';
    line.dataset.id=id;
    line.style.left=cx+'px';
    line.style.top=cy+'px';
    line.style.width=len+'px';
    line.style.transform=`rotate(${Math.atan2(dy,dx)}rad)`;
    imageBox.appendChild(line);
  }

  items.forEach((item,idx)=>{
    const [bx1,by1,bx2,by2]=smartHotspotBbox(item);
    const x1=bx1*sx, y1=by1*sy, x2=bx2*sx, y2=by2*sy;
    const cx=(x1+x2)/2, cy=(y1+y2)/2;
    const labelText=`${item.display_name}${item.count>1?' ×'+item.count:''}`;

    const h=document.createElement('a');
    h.className='hotspot';
    h.dataset.id=item.id;
    h.setAttribute('aria-label', item.display_name);
    h.href=item.link;
    h.target='_blank';
    h.rel='noopener';
    h.style.left=x1+'px';
    h.style.top=y1+'px';
    h.style.width=Math.max(12,x2-x1)+'px';
    h.style.height=Math.max(12,y2-y1)+'px';
    h.style.zIndex=String(10+idx);
    h.addEventListener('mouseenter',()=>activate(item.id));
    h.addEventListener('mouseleave',()=>activate(null));
    imageBox.appendChild(h);

    const r=bestLabelRect(cx,cy,x1,y1,x2,y2,labelText);
    addLeader(item.id,x1,y1,x2,y2,r.x,r.y,r.w,r.h);

    const lab=document.createElement('a');
    lab.className='hotspot-label';
    lab.dataset.id=item.id;
    lab.href=item.link;
    lab.target='_blank';
    lab.rel='noopener';
    lab.setAttribute('aria-label', `Open Amazon search for ${item.display_name}`);
    lab.textContent=labelText;
    lab.style.left=r.x+'px';
    lab.style.top=r.y+'px';
    lab.style.width=r.w+'px';
    lab.addEventListener('mouseenter',()=>activate(item.id));
    lab.addEventListener('mouseleave',()=>activate(null));
    imageBox.appendChild(lab);
  });
}
function displayItems(){
  const raw=(lastData?.items||[]);
  const hasConferenceChair=raw.some(i=>String(i.display_name||'').toLowerCase().includes('conference chair'));
  const whiteboards=raw.filter(i=>String(i.display_name||'').toLowerCase().includes('whiteboard'));
  const cleaned=[];

  function bboxCenter(item){const [x1,y1,x2,y2]=item.bbox.map(Number); return [(x1+x2)/2,(y1+y2)/2];}
  function inside(item,boxItem,pad=18){
    const [cx,cy]=bboxCenter(item); const [x1,y1,x2,y2]=boxItem.bbox.map(Number);
    return cx>=x1-pad && cx<=x2+pad && cy>=y1-pad && cy<=y2+pad;
  }
  function overlapRatio(a,b){
    const [ax1,ay1,ax2,ay2]=a.bbox.map(Number), [bx1,by1,bx2,by2]=b.bbox.map(Number);
    const ix=Math.max(0,Math.min(ax2,bx2)-Math.max(ax1,bx1));
    const iy=Math.max(0,Math.min(ay2,by2)-Math.max(ay1,by1));
    const inter=ix*iy;
    const aa=Math.max(1,(ax2-ax1)*(ay2-ay1));
    return inter/aa;
  }

  raw.forEach(i=>{
    const n=String(i.name||'').toLowerCase();
    const d=String(i.display_name||'').toLowerCase();

    // remove labels the user explicitly does not want in the web UI
    if(n.includes('linear slot diffuser')||d.includes('linear slot diffuser')||n.includes('ceiling linear diffuser')) return;

    // if Conference Chair exists, hide generic duplicate Chair
    if(hasConferenceChair && (d==='chair'||n==='chair')) return;

    // YOLO sometimes calls the whiteboard a Monitor / Screen. Hide that false duplicate.
    const isMonitor=d.includes('monitor') || (d.includes('screen') && !d.includes('television')) || n==='monitor' || n==='screen' || n==='computer monitor';
    if(isMonitor && whiteboards.some(w=>inside(i,w,28)||overlapRatio(i,w)>0.10)) return;

    cleaned.push({...i});
  });

  // Merge exact duplicate display labels (example: Lamp + Lamp => Lamp ×2)
  const merged=new Map();
  cleaned.forEach(i=>{
    const key=String(i.display_name||i.name).toLowerCase()+'|'+String(i.category||'').toLowerCase();
    if(!merged.has(key)){ merged.set(key,{...i,count:Number(i.count||1)}); return; }
    const prev=merged.get(key);
    prev.count=Number(prev.count||1)+Number(i.count||1);
    // keep bbox/link/confidence from the stronger detection so hover still points cleanly
    if(Number(i.confidence||0)>Number(prev.confidence||0)){
      const keepCount=prev.count;
      merged.set(key,{...i,count:keepCount});
    }
  });
  return Array.from(merged.values());
}

function categories(){ const s=new Set(displayItems().map(i=>i.category)); return ['All',...Array.from(s).sort()]; }
function renderChips(){ const wrap=$('#chips'); wrap.innerHTML=''; categories().forEach(cat=>{ const b=document.createElement('button'); b.className='chip'+(cat===activeCat?' active':''); b.dataset.cat=cat; b.textContent=cat; b.onclick=()=>{activeCat=cat; renderChips(); renderItems(); renderImage();}; wrap.appendChild(b); }); }
function filteredItems(){ const q=$('#search').value.trim().toLowerCase(); return displayItems().filter(i=>(activeCat==='All'||i.category===activeCat)&&(!q||i.display_name.toLowerCase().includes(q)||i.category.toLowerCase().includes(q)||i.name.toLowerCase().includes(q)));  }
$('#search').addEventListener('input',()=>{renderItems(); renderImage();});
function renderItems(){ const list=$('#items'); const arr=filteredItems(); if(!lastData){return} if(!arr.length){ list.innerHTML='<p class="mini">No items match this filter.</p>'; return;} list.innerHTML=''; arr.forEach(item=>{ const card=document.createElement('div'); card.className='card'; card.dataset.id=item.id; card.innerHTML=`<div><h3>${escapeHtml(item.display_name)} ${item.count>1?`×${item.count}`:''}</h3><div class="meta"><span class="tag">${escapeHtml(item.category)}</span><span class="tag">Detected</span><span class="tag">${item.bbox.join(', ')}</span></div></div><a class="buy" href="${item.link}" target="_blank" rel="noopener">Buy</a>`; card.addEventListener('mouseenter',()=>activate(item.id)); card.addEventListener('mouseleave',()=>activate(null)); card.addEventListener('click',e=>{ if(e.target.tagName!=='A'){ activeId=item.id; activate(item.id,true); }}); list.appendChild(card); }); }
function activate(id,persist=false){ if(!persist && activeId) id=activeId; $$('.card,.hotspot,.hotspot-label,.leader-line').forEach(el=>el.classList.toggle('active',!!id&&el.dataset.id===id)); }
function escapeHtml(str){return String(str).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function download(filename,content,type='application/octet-stream'){ const blob=content instanceof Blob?content:new Blob([content],{type}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=filename; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000); }
$('#exportJson').onclick=()=>{ if(!lastData)return toast('Detect first'); download('furniture-detection-report.json',JSON.stringify(lastData,null,2),'application/json') };
$('#exportCsv').onclick=()=>{ if(!lastData)return toast('Detect first'); const rows=[['Name','Category','Confidence','Count','BBox','Amazon Link'],...lastData.items.map(i=>[i.display_name,i.category,i.confidence_percent,i.count,i.bbox.join(' '),i.link])]; const csv=rows.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\n'); download('furniture-detection-report.csv',csv,'text/csv') };
$('#copyLinks').onclick=async()=>{ if(!lastData)return toast('Detect first'); await navigator.clipboard.writeText(lastData.items.map(i=>`${i.display_name}: ${i.link}`).join('\n')); toast('Links copied'); };
$('#downloadImg').onclick=()=>{ if(!lastData)return toast('Detect first'); fetch(lastData.image).then(r=>r.blob()).then(b=>download('detected-furniture.jpg',b,'image/jpeg')); };
function setZoom(z){ zoom=Math.max(50,Math.min(200,z)); $('#zoomRange').value=zoom; $('#zoomLabel').textContent=zoom+'%'; stage.style.transform=`scale(${zoom/100})`; }
$('#zoomRange').oninput=e=>setZoom(+e.target.value); $('#zoomIn').onclick=()=>setZoom(zoom+10); $('#zoomOut').onclick=()=>setZoom(zoom-10); $('#fitBtn').onclick=()=>setZoom(100);
$('#resetBtn').onclick=()=>{ selectedFile=null; lastData=null; fileInput.value=''; detectBtn.disabled=true; mainImage.removeAttribute('src'); clearHotspots(); stage.classList.add('hidden'); empty.classList.remove('hidden'); $('#filePill').textContent='No image selected'; $('#sizePill').textContent='—'; $('#speedPill').textContent='—'; $('#uniqueStat').textContent='0'; $('#totalStat').textContent='0'; $('#confStat').textContent='0%'; $('#catStat').textContent='—'; $('#chips').innerHTML='<button class="chip active" data-cat="All">All</button>'; $('#items').innerHTML='<p class="mini">Detected objects will appear here. Hover a card to highlight the object; click Buy to open product search.</p>'; toast('Reset done'); };
$('#demoTips').onclick=()=>toast('Use clear room photos. Try confidence 10–18% for more tags, 25%+ for cleaner tags.');
window.addEventListener('resize',()=>{ if(lastData) renderImage(); });
</script>
</body>
</html>
'''


def find_free_port(start_port: int = 7860, max_tries: int = 50) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


if __name__ == "__main__":
    import uvicorn

    port = find_free_port(7860)
    print(f"\nFurniVision AI running at: http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port)
