from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _to_out(t: models.Task) -> schemas.TaskOut:
    out = schemas.TaskOut.model_validate(t)
    out.assignee_name = t.assignee.name if t.assignee else None
    return out


@router.get("", response_model=list[schemas.TaskOut])
def list_tasks(db: Session = Depends(get_db)):
    tasks = (
        db.query(models.Task)
        .order_by(models.Task.order, models.Task.created_at)
        .all()
    )
    return [_to_out(t) for t in tasks]


@router.post("", response_model=schemas.TaskOut, status_code=201)
def create_task(payload: schemas.TaskIn, db: Session = Depends(get_db)):
    t = models.Task(**payload.model_dump())
    db.add(t)
    db.commit()
    db.refresh(t)
    return _to_out(t)


@router.put("/{task_id}", response_model=schemas.TaskOut)
def update_task(task_id: int, payload: schemas.TaskIn, db: Session = Depends(get_db)):
    t = db.get(models.Task, task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    for k, v in payload.model_dump().items():
        setattr(t, k, v)
    db.commit()
    db.refresh(t)
    return _to_out(t)


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    t = db.get(models.Task, task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    db.delete(t)
    db.commit()
