from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/models3d", tags=["models3d"])


@router.get("", response_model=list[schemas.Model3DOut])
def list_models(db: Session = Depends(get_db)):
    return db.query(models.Model3D).order_by(models.Model3D.name).all()


@router.post("", response_model=schemas.Model3DOut, status_code=201)
def create_model(payload: schemas.Model3DIn, db: Session = Depends(get_db)):
    m = models.Model3D(**payload.model_dump())
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


@router.put("/{model_id}", response_model=schemas.Model3DOut)
def update_model(model_id: int, payload: schemas.Model3DIn, db: Session = Depends(get_db)):
    m = db.get(models.Model3D, model_id)
    if not m:
        raise HTTPException(404, "3D model not found")
    for k, v in payload.model_dump().items():
        setattr(m, k, v)
    db.commit()
    db.refresh(m)
    return m


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    m = db.get(models.Model3D, model_id)
    if not m:
        raise HTTPException(404, "3D model not found")
    db.delete(m)
    db.commit()
