# Mouse-Arducam · Project Management App

A self-hosted web app to manage every aspect of the 10-camera mouse-tracking rig:
tasks & assignments, crystal-clear handoff docs, **calibration test tracking with drift
charts**, an equipment/cost list, 3D-print tracking, and a one-click runner for the
recording/solve scripts.

- **Backend:** FastAPI + SQLite (SQLAlchemy). Single-file database, no external services.
- **Frontend:** dependency-free vanilla JS (no build step, no CDN — works fully offline).
- **Isolation:** its own `pm/.venv`, completely separate from the vision project's `.venv`.

---

## Quick start (on the mini PC)

```bash
cd /home/cat/mouse-arducam/pm
./run.sh
```

Then open <http://localhost:8000>. On first run the database is created and **seeded with
the real project state** (the 3 documented extrinsic solves, equipment, 3D parts, and the
open calibration tasks) so the charts and boards are useful immediately.

Change the port: `PM_PORT=9000 ./run.sh`

---

## Requiring a password (do this before remote access)

The app is unauthenticated by default. To gate it behind a shared team password, set
`PM_PASSWORD` before launching:

```bash
PM_PASSWORD='choose-a-strong-shared-password' ./run.sh
```

Everyone then logs in at `/login` once; the session cookie lasts 30 days. The API returns
`401` without it. (The password is compared with a constant-time check; the cookie is
HMAC-signed with a per-process secret.)

---

## Remote team access

The server binds `0.0.0.0`, so it's reachable on the LAN at `http://<mini-pc-ip>:8000`.
For teammates **outside** the LAN, pick one of these:

### Option A — Tailscale (recommended: private, free, no public exposure)

1. Install on the mini PC: `curl -fsSL https://tailscale.com/install.sh | sh`
2. `sudo tailscale up` and sign in.
3. Invite teammates to your tailnet; they install Tailscale too.
4. They open `http://<mini-pc-tailscale-ip>:8000` (or the MagicDNS name).

Nothing is exposed to the public internet — only your tailnet can reach it.

### Option B — Cloudflare Tunnel (public HTTPS URL)

1. Install `cloudflared` on the mini PC.
2. `cloudflared tunnel --url http://localhost:8000`
3. Share the generated `https://…trycloudflare.com` URL.

**Always set `PM_PASSWORD` when using a public tunnel.**

### Option C — Plain LAN only

Just share `http://<mini-pc-ip>:8000`. Find the IP with `hostname -I`.

---

## Run it as a background service (optional)

So it survives logout / reboot, create `/etc/systemd/system/mouse-pm.service`:

```ini
[Unit]
Description=Mouse-Arducam PM app
After=network.target

[Service]
User=cat
WorkingDirectory=/home/cat/mouse-arducam/pm
Environment=PM_PASSWORD=your-shared-password
ExecStart=/home/cat/mouse-arducam/pm/run.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then: `sudo systemctl enable --now mouse-pm`.

---

## Data, backup, reset

- All data lives in **`pm/data/pm.db`** (git-ignored). Back it up by copying that file.
- Script run logs are in `pm/data/logs/`.
- **Reset to a fresh seeded state:** stop the app, `rm pm/data/pm.db`, start again.

---

## The "Run Scripts" tab — safety notes

- Only an **allowlist** of scripts can be launched (see `app/routers/scripts.py`):
  record intrinsic / extrinsic / reconstruction, and the headless extrinsic solve.
- You may edit a script's **arguments**, but never *which program* runs — no shell is used.
- Scripts run with the **vision project's `.venv`** (`/home/cat/mouse-arducam/.venv`),
  which has OpenCV/Caliscope. Recording scripts open the physical cameras.
- Because this can drive real hardware, **set `PM_PASSWORD` whenever the app is reachable
  beyond your own machine.**

---

## Project layout

```
pm/
  run.sh                 # create venv + launch
  requirements.txt
  app/
    main.py              # FastAPI app, auth gate, static serving
    database.py          # SQLite engine/session
    models.py            # ORM tables
    schemas.py           # Pydantic models
    seed.py              # initial data from documentation.md / 9Jul.md
    thresholds.py        # pass/warn/fail computation
    routers/             # team, tasks, calibration, equipment, models3d, settings, docs, scripts
  static/                # index.html + css + js (SPA)
  docs/onboarding.md     # in-app handoff guide
  data/                  # pm.db + logs (git-ignored)
```

## API (for reference / scripting)

`GET/POST/PUT/DELETE` on: `/api/team`, `/api/tasks`, `/api/calibration`,
`/api/equipment`, `/api/models3d`. Also `GET/PUT /api/settings`,
`GET /api/docs`, and the scripts endpoints under `/api/scripts`.
Interactive docs at `/docs` (FastAPI Swagger UI).
