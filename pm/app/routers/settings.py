from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..thresholds import get_thresholds

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def read_settings(db: Session = Depends(get_db)):
    """Return effective settings (defaults merged with stored overrides)."""
    return get_thresholds(db)


@router.put("/{key}")
def set_setting(key: str, payload: schemas.SettingIn, db: Session = Depends(get_db)):
    s = db.get(models.Setting, key)
    if s:
        s.value = payload.value
    else:
        db.add(models.Setting(key=key, value=payload.value))
    db.commit()
    return get_thresholds(db)
