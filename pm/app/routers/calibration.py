from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..thresholds import compute_status, get_thresholds

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


def _to_out(run: models.CalibrationRun, th: dict) -> schemas.CalibrationRunOut:
    out = schemas.CalibrationRunOut.model_validate(run)
    out.status = compute_status(run, th)
    return out


@router.get("", response_model=list[schemas.CalibrationRunOut])
def list_runs(db: Session = Depends(get_db)):
    th = get_thresholds(db)
    runs = (
        db.query(models.CalibrationRun)
        .order_by(models.CalibrationRun.run_date, models.CalibrationRun.id)
        .all()
    )
    return [_to_out(r, th) for r in runs]


@router.post("", response_model=schemas.CalibrationRunOut, status_code=201)
def create_run(payload: schemas.CalibrationRunIn, db: Session = Depends(get_db)):
    r = models.CalibrationRun(**payload.model_dump())
    db.add(r)
    db.commit()
    db.refresh(r)
    return _to_out(r, get_thresholds(db))


@router.put("/{run_id}", response_model=schemas.CalibrationRunOut)
def update_run(run_id: int, payload: schemas.CalibrationRunIn, db: Session = Depends(get_db)):
    r = db.get(models.CalibrationRun, run_id)
    if not r:
        raise HTTPException(404, "Calibration run not found")
    for k, v in payload.model_dump().items():
        setattr(r, k, v)
    db.commit()
    db.refresh(r)
    return _to_out(r, get_thresholds(db))


@router.delete("/{run_id}", status_code=204)
def delete_run(run_id: int, db: Session = Depends(get_db)):
    r = db.get(models.CalibrationRun, run_id)
    if not r:
        raise HTTPException(404, "Calibration run not found")
    db.delete(r)
    db.commit()
