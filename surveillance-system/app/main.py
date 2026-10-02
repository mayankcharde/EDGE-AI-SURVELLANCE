# import os
# import logging
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

# from config import BASE_DIR, SNAPSHOT_DIR, CLIP_DIR
# from app.database import init_db, get_events, get_event, get_stats

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )

# app = FastAPI(title="Surveillance API")

# templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
# app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
# if os.path.isdir(CLIP_DIR):
#     app.mount("/clips", StaticFiles(directory=CLIP_DIR), name="clips")


# @app.on_event("startup")
# def startup():
#     init_db()


# @app.get("/", response_class=HTMLResponse)
# def dashboard(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.get("/api/events")
# def api_events(limit: int = 100, event: str = None,
#                severity: str = None, person: str = None):
#     return get_events(limit=limit, event_type=event, severity=severity, person=person)


# @app.get("/api/events/{event_id}")
# def api_event(event_id: int):
#     ev = get_event(event_id)
#     if not ev:
#         raise HTTPException(status_code=404, detail="Event not found")
#     return ev


# @app.get("/api/stats")
# def api_stats():
#     return get_stats()


# @app.get("/api/health")
# def health():
#     return {"status": "ok"}





# import os
# import shutil
# import logging
# from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

# from config import (
#     BASE_DIR, SNAPSHOT_DIR, CLIP_DIR, UPLOAD_DIR,
#     SOURCE_TYPE, WEBCAM_INDEX, RTSP_URL
# )
# from app.database import init_db, get_events, get_event, get_stats

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )

# app = FastAPI(title="Surveillance API")

# templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
# app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
# if os.path.isdir(CLIP_DIR):
#     app.mount("/clips", StaticFiles(directory=CLIP_DIR), name="clips")

# # Holds the currently running pipeline (set by run.py)
# pipeline_ref = {"instance": None, "current_source": None}


# @app.on_event("startup")
# def startup():
#     init_db()


# @app.get("/", response_class=HTMLResponse)
# def dashboard(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# # ---------------- Events API ----------------
# @app.get("/api/events")
# def api_events(limit: int = 100, event: str = None,
#                severity: str = None, person: str = None):
#     return get_events(limit=limit, event_type=event, severity=severity, person=person)


# @app.get("/api/events/{event_id}")
# def api_event(event_id: int):
#     ev = get_event(event_id)
#     if not ev:
#         raise HTTPException(404, "Event not found")
#     return ev


# @app.get("/api/stats")
# def api_stats():
#     return get_stats()


# @app.get("/api/health")
# def health():
#     return {"status": "ok"}


# # ---------------- Source control ----------------
# @app.get("/api/source")
# def api_source():
#     p = pipeline_ref["instance"]
#     info = {
#         "current": pipeline_ref["current_source"],
#         "config_type": SOURCE_TYPE,
#         "config_rtsp": RTSP_URL,
#         "config_webcam": WEBCAM_INDEX,
#     }
#     if p and p.camera.is_file:
#         info["progress"] = p.camera.get_progress()
#     return info


# @app.post("/api/source/webcam")
# def use_webcam(index: int = Form(0)):
#     p = pipeline_ref["instance"]
#     if not p:
#         raise HTTPException(500, "Pipeline not running")
#     p.switch_source(int(index), is_file=False)
#     pipeline_ref["current_source"] = f"webcam:{index}"
#     return {"status": "ok", "source": f"webcam:{index}"}


# @app.post("/api/source/rtsp")
# def use_rtsp(url: str = Form(...)):
#     p = pipeline_ref["instance"]
#     if not p:
#         raise HTTPException(500, "Pipeline not running")
#     p.switch_source(url, is_file=False)
#     pipeline_ref["current_source"] = f"rtsp:{url}"
#     return {"status": "ok", "source": url}


# @app.post("/api/source/video")
# async def use_video(file: UploadFile = File(...)):
#     """Accept a demo video upload and switch pipeline to it."""
#     p = pipeline_ref["instance"]
#     if not p:
#         raise HTTPException(500, "Pipeline not running")

#     # Save upload
#     safe_name = file.filename.replace(" ", "_")
#     dest = os.path.join(UPLOAD_DIR, safe_name)
#     with open(dest, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     p.switch_source(dest, is_file=True)
#     pipeline_ref["current_source"] = f"video:{safe_name}"
#     return {"status": "ok", "source": safe_name, "path": dest}





import os
import logging
import webbrowser
import threading
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    BASE_DIR, SNAPSHOT_DIR, CLIP_DIR, UPLOAD_DIR,
    SOURCE_TYPE, WEBCAM_INDEX, RTSP_URL
)
from app.database import init_db, get_events, get_event, get_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="Surveillance API")

# ---------------- Templates + Static ----------------
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

logger = logging.getLogger(__name__)
logger.info(f"Templates dir: {TEMPLATES_DIR} (exists={os.path.isdir(TEMPLATES_DIR)})")
logger.info(f"Static dir:    {STATIC_DIR} (exists={os.path.isdir(STATIC_DIR)})")

if not os.path.isfile(os.path.join(TEMPLATES_DIR, "index.html")):
    logger.error(f"templates/index.html NOT FOUND at {TEMPLATES_DIR}/index.html")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.isdir(SNAPSHOT_DIR):
    app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
if os.path.isdir(CLIP_DIR):
    app.mount("/clips", StaticFiles(directory=CLIP_DIR), name="clips")

pipeline_ref = {"instance": None, "current_source": None}


@app.on_event("startup")
def startup():
    init_db()


# ---------------- Dashboard ----------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_alias(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ---------------- Events API ----------------
@app.get("/api/events")
def api_events(limit: int = 100, event: str = None,
               severity: str = None, person: str = None):
    return get_events(limit=limit, event_type=event, severity=severity, person=person)


@app.get("/api/events/{event_id}")
def api_event(event_id: int):
    ev = get_event(event_id)
    if not ev:
        raise HTTPException(404, "Event not found")
    return ev


@app.get("/api/stats")
def api_stats():
    return get_stats()


@app.get("/api/health")
def health():
    return {"status": "ok"}


from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
import cv2
import time

@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            p = pipeline_ref.get("instance")
            if p and getattr(p, "latest_frame", None) is not None:
                ret, buffer = cv2.imencode('.jpg', p.latest_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/source")
def api_source():
    p = pipeline_ref["instance"]
    info = {
        "current": pipeline_ref["current_source"],
        "config_type": SOURCE_TYPE,
        "config_rtsp": RTSP_URL,
        "config_webcam": WEBCAM_INDEX,
    }
    if p and p.camera.is_file:
        info["progress"] = p.camera.get_progress()
    return info


@app.post("/api/source/webcam")
def use_webcam(index: int = Form(0)):
    p = pipeline_ref["instance"]
    if not p:
        raise HTTPException(500, "Pipeline not running")
    p.switch_source(int(index), is_file=False)
    pipeline_ref["current_source"] = f"webcam:{index}"
    return {"status": "ok", "source": f"webcam:{index}"}


@app.post("/api/source/rtsp")
def use_rtsp(url: str = Form(...)):
    p = pipeline_ref["instance"]
    if not p:
        raise HTTPException(500, "Pipeline not running")
    p.switch_source(url, is_file=False)
    pipeline_ref["current_source"] = f"rtsp:{url}"
    return {"status": "ok", "source": url}


@app.post("/api/source/video")
async def use_video(file: UploadFile = File(...)):
    p = pipeline_ref["instance"]
    if not p:
        raise HTTPException(500, "Pipeline not running")

    safe_name = file.filename.replace(" ", "_")
    dest = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    p.switch_source(dest, is_file=True)
    pipeline_ref["current_source"] = f"video:{safe_name}"
    return {"status": "ok", "source": safe_name, "path": dest}