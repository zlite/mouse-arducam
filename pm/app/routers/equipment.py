from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/equipment", tags=["equipment"])


@router.get("", response_model=list[schemas.EquipmentOut])
def list_equipment(db: Session = Depends(get_db)):
    return db.query(models.Equipment).order_by(models.Equipment.category, models.Equipment.name).all()


@router.post("", response_model=schemas.EquipmentOut, status_code=201)
def create_equipment(payload: schemas.EquipmentIn, db: Session = Depends(get_db)):
    e = models.Equipment(**payload.model_dump())
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


@router.put("/{item_id}", response_model=schemas.EquipmentOut)
def update_equipment(item_id: int, payload: schemas.EquipmentIn, db: Session = Depends(get_db)):
    e = db.get(models.Equipment, item_id)
    if not e:
        raise HTTPException(404, "Equipment item not found")
    for k, v in payload.model_dump().items():
        setattr(e, k, v)
    db.commit()
    db.refresh(e)
    return e


@router.delete("/{item_id}", status_code=204)
def delete_equipment(item_id: int, db: Session = Depends(get_db)):
    e = db.get(models.Equipment, item_id)
    if not e:
        raise HTTPException(404, "Equipment item not found")
    db.delete(e)
    db.commit()
