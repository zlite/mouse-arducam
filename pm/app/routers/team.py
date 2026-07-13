from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/team", tags=["team"])


@router.get("", response_model=list[schemas.TeamMemberOut])
def list_members(db: Session = Depends(get_db)):
    return db.query(models.TeamMember).order_by(models.TeamMember.name).all()


@router.post("", response_model=schemas.TeamMemberOut, status_code=201)
def create_member(payload: schemas.TeamMemberIn, db: Session = Depends(get_db)):
    m = models.TeamMember(**payload.model_dump())
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


@router.put("/{member_id}", response_model=schemas.TeamMemberOut)
def update_member(member_id: int, payload: schemas.TeamMemberIn, db: Session = Depends(get_db)):
    m = db.get(models.TeamMember, member_id)
    if not m:
        raise HTTPException(404, "Team member not found")
    for k, v in payload.model_dump().items():
        setattr(m, k, v)
    db.commit()
    db.refresh(m)
    return m


@router.delete("/{member_id}", status_code=204)
def delete_member(member_id: int, db: Session = Depends(get_db)):
    m = db.get(models.TeamMember, member_id)
    if not m:
        raise HTTPException(404, "Team member not found")
    db.delete(m)
    db.commit()
