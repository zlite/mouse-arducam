"""ORM models for the project-management app.

Tables:
  TeamMember      - people who work on the project (assignees)
  Task            - work items on a kanban board
  CalibrationRun  - a single calibration test result (drift tracking)
  Equipment       - hardware / consumables with cost
  Model3D         - parts that need to be 3D printed
  ScriptRun       - a launched calibration/utility script and its live status
  Setting         - key/value config (calibration thresholds, project title)
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TeamMember(Base):
    __tablename__ = "team_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(120), default="")
    email: Mapped[str] = mapped_column(String(200), default="")
    active: Mapped[bool] = mapped_column(default=True)

    tasks: Mapped[list["Task"]] = relationship(back_populates="assignee")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    # Hardware / Calibration / Software / Docs / Procurement / 3D-Printing / Data
    category: Mapped[str] = mapped_column(String(60), default="Calibration")
    # todo / in_progress / blocked / done
    status: Mapped[str] = mapped_column(String(30), default="todo")
    # low / medium / high / critical
    priority: Mapped[str] = mapped_column(String(20), default="medium")
    assignee_id: Mapped[int | None] = mapped_column(
        ForeignKey("team_members.id", ondelete="SET NULL"), nullable=True
    )
    due_date: Mapped[str | None] = mapped_column(String(20), nullable=True)  # YYYY-MM-DD
    order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    assignee: Mapped["TeamMember | None"] = relationship(back_populates="tasks")


class CalibrationRun(Base):
    __tablename__ = "calibration_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_date: Mapped[str] = mapped_column(String(20))  # YYYY-MM-DD
    # intrinsic / extrinsic / reconstruction
    type: Mapped[str] = mapped_column(String(30), default="extrinsic")
    label: Mapped[str] = mapped_column(String(200), default="")
    reprojection_rmse_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    volumetric_scale_rmse_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_observations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_cameras: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # {"cam_0": 33.2, ...}
    per_camera_rmse: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Equipment(Base):
    __tablename__ = "equipment"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(80), default="")
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_cost: Mapped[float | None] = mapped_column(Float, nullable=True)  # None = TBD
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    supplier: Mapped[str] = mapped_column(String(200), default="")
    url: Mapped[str] = mapped_column(String(500), default="")
    # owned / ordered / needed
    status: Mapped[str] = mapped_column(String(20), default="owned")
    notes: Mapped[str] = mapped_column(Text, default="")


class Model3D(Base):
    __tablename__ = "models_3d"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    purpose: Mapped[str] = mapped_column(Text, default="")
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    material: Mapped[str] = mapped_column(String(80), default="PLA")
    file_link: Mapped[str] = mapped_column(String(500), default="")
    # not_started / printing / printed / failed
    print_status: Mapped[str] = mapped_column(String(20), default="not_started")
    printed_by: Mapped[str] = mapped_column(String(120), default="")
    notes: Mapped[str] = mapped_column(Text, default="")


class ScriptRun(Base):
    __tablename__ = "script_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    script_id: Mapped[str] = mapped_column(String(80))
    script_name: Mapped[str] = mapped_column(String(200))
    command: Mapped[str] = mapped_column(Text, default="")
    # running / finished / failed / stopped
    status: Mapped[str] = mapped_column(String(20), default="running")
    return_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    log_path: Mapped[str] = mapped_column(String(500), default="")
    started_by: Mapped[str] = mapped_column(String(120), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
