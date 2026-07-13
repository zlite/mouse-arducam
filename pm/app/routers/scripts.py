"""Launch and monitor project scripts on the rig host.

SAFETY: only scripts defined in app/scripts_registry.py can be run. The executable
and its base arguments come from that whitelist; the user may only *append* extra
arguments (parsed with shlex, never a shell). This is an internal tool for a trusted
team — protect it with PM_PASSWORD when exposed off-LAN.
"""
import shlex
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..scripts_registry import REPO_ROOT, SCRIPTS, SCRIPTS_BY_ID, command_string

router = APIRouter(prefix="/api/scripts", tags=["scripts"])

PM_DIR = Path(__file__).resolve().parent.parent.parent  # .../pm
LOG_DIR = PM_DIR / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# In-memory registry of live processes: {script_run_id: Popen}
_PROCS: dict[int, subprocess.Popen] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _reconcile(db: Session) -> None:
    """Update DB status for runs whose process has exited or become untracked."""
    running = db.query(models.ScriptRun).filter(models.ScriptRun.status == "running").all()
    changed = False
    for run in running:
        proc = _PROCS.get(run.id)
        if proc is None:
            run.status = "unknown"  # server likely restarted; can no longer track
            run.finished_at = _now()
            changed = True
            continue
        rc = proc.poll()
        if rc is not None:
            run.return_code = rc
            run.status = "finished" if rc == 0 else "failed"
            run.finished_at = _now()
            _PROCS.pop(run.id, None)
            changed = True
    if changed:
        db.commit()


@router.get("")
def list_catalog():
    """Return the whitelisted, runnable scripts."""
    return [
        {
            "script_id": s["id"],
            "name": s["name"],
            "category": s.get("category", ""),
            "description": s["description"],
            "base_command": command_string(s),
            "gui": s.get("gui", False),
            "long_running": s.get("long_running", False),
        }
        for s in SCRIPTS
    ]


@router.get("/runs", response_model=list[schemas.ScriptRunOut])
def list_runs(db: Session = Depends(get_db)):
    _reconcile(db)
    return (
        db.query(models.ScriptRun)
        .order_by(models.ScriptRun.started_at.desc())
        .limit(100)
        .all()
    )


@router.get("/runs/{run_id}", response_model=schemas.ScriptRunDetail)
def get_run(run_id: int, db: Session = Depends(get_db)):
    _reconcile(db)
    run = db.get(models.ScriptRun, run_id)
    if not run:
        raise HTTPException(404, "Script run not found")
    detail = schemas.ScriptRunDetail.model_validate(run)
    if run.log_path and Path(run.log_path).exists():
        detail.output = Path(run.log_path).read_text(encoding="utf-8", errors="replace")[-20000:]
    return detail


@router.post("/runs/{run_id}/stop", response_model=schemas.ScriptRunOut)
def stop_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(models.ScriptRun, run_id)
    if not run:
        raise HTTPException(404, "Script run not found")
    proc = _PROCS.get(run_id)
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    run.status = "stopped"
    run.finished_at = _now()
    run.return_code = proc.returncode if proc else None
    _PROCS.pop(run_id, None)
    db.commit()
    db.refresh(run)
    return run


@router.post("/{script_id}/run", response_model=schemas.ScriptRunOut, status_code=201)
def run_script(script_id: str, payload: schemas.ScriptRunRequest, db: Session = Depends(get_db)):
    script = SCRIPTS_BY_ID.get(script_id)
    if not script:
        raise HTTPException(404, "Unknown script")

    base_argv = list(script["command"])
    # base_argv[1] is the script file (base_argv[0] is the interpreter / bash).
    target = REPO_ROOT / base_argv[1] if len(base_argv) > 1 else None
    if target and not target.exists():
        raise HTTPException(400, f"Script file not found: {base_argv[1]}")

    try:
        extra_argv = shlex.split(payload.args) if payload.args.strip() else []
    except ValueError as exc:
        raise HTTPException(400, f"Could not parse arguments: {exc}")

    argv = [*base_argv, *extra_argv]
    command_str = " ".join(shlex.quote(a) for a in argv)

    run = models.ScriptRun(
        script_id=script_id,
        script_name=script["name"],
        command=command_str,
        status="running",
        started_by=payload.started_by or "unknown",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    log_path = LOG_DIR / f"run_{run.id}.log"
    run.log_path = str(log_path)
    db.commit()

    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"$ {command_str}\n(cwd: {REPO_ROOT})\n\n")
    log_file.flush()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        log_file.write(f"\n[launch error] {exc}\n")
        log_file.close()
        run.status = "failed"
        run.finished_at = _now()
        db.commit()
        raise HTTPException(500, f"Failed to launch: {exc}")

    _PROCS[run.id] = proc
    db.refresh(run)
    return run
