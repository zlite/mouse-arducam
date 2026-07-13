"""FastAPI entry point for the Mouse-Arducam project-management app.

- Creates tables and seeds initial data on startup.
- Serves the JSON API under /api/* and the SPA from /static.
- Optional shared-password gate: set env var PM_PASSWORD to require login
  (recommended when the app is reachable off-LAN).
"""
import hashlib
import hmac
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .database import Base, SessionLocal, engine
from .routers import (
    calibration,
    docs,
    equipment,
    models3d,
    scripts,
    settings,
    tasks,
    team,
)
from .seed import seed_all

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

PM_PASSWORD = os.environ.get("PM_PASSWORD", "").strip()
# Stable-per-process secret used to sign the auth cookie.
_COOKIE_SECRET = secrets.token_hex(16)
COOKIE_NAME = "pm_auth"


def _expected_token() -> str:
    return hmac.new(_COOKIE_SECRET.encode(), PM_PASSWORD.encode(), hashlib.sha256).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_all(db)
    finally:
        db.close()
    if not PM_PASSWORD:
        print("\n[pm] WARNING: PM_PASSWORD not set — the app is UNAUTHENTICATED.")
        print("[pm] Set PM_PASSWORD before exposing it beyond localhost/LAN.\n")
    yield


app = FastAPI(title="Mouse-Arducam PM", lifespan=lifespan)


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    # No password configured -> open access (dev / LAN).
    if not PM_PASSWORD:
        return await call_next(request)

    path = request.url.path
    open_paths = ("/login", "/static/css", "/static/js", "/favicon.ico", "/health")
    if path.startswith(open_paths):
        return await call_next(request)

    token = request.cookies.get(COOKIE_NAME, "")
    if hmac.compare_digest(token, _expected_token()):
        return await call_next(request)

    if path.startswith("/api/"):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=302)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login — Mouse-Arducam PM</title>
<style>body{font-family:system-ui,sans-serif;background:#0f1419;color:#e6e6e6;
display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}
form{background:#1a2230;padding:2rem;border-radius:12px;width:280px}
input,button{width:100%;box-sizing:border-box;padding:.6rem;margin-top:.6rem;
border-radius:8px;border:1px solid #33415a;background:#0f1419;color:#e6e6e6}
button{background:#3b82f6;border:none;font-weight:600;cursor:pointer;margin-top:1rem}
h2{margin:0 0 .5rem}</style></head><body>
<form method="post" action="/login">
<h2>🐭 Project Login</h2>
<div style="color:#8b98ad;font-size:.85rem">Enter the shared team password.</div>
<input type="password" name="password" placeholder="Password" autofocus>
<button type="submit">Sign in</button>
</form></body></html>"""


@app.post("/login")
def login(password: str = Form(...)):
    if not hmac.compare_digest(password, PM_PASSWORD):
        raise HTTPException(401, "Wrong password")
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie(
        COOKIE_NAME, _expected_token(), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30
    )
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(COOKIE_NAME)
    return resp


# API routers
for r in (team, tasks, calibration, equipment, models3d, settings, docs, scripts):
    app.include_router(r.router)


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# Static assets (mounted last so it doesn't shadow /api).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
