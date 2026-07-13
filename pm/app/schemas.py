"""Pydantic request/response schemas."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict


# ---------- Team ----------
class TeamMemberIn(BaseModel):
    name: str
    role: str = ""
    email: str = ""
    active: bool = True


class TeamMemberOut(TeamMemberIn):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- Tasks ----------
class TaskIn(BaseModel):
    title: str
    description: str = ""
    category: str = "Calibration"
    status: str = "todo"
    priority: str = "medium"
    assignee_id: int | None = None
    due_date: str | None = None
    order: int = 0


class TaskOut(TaskIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    updated_at: datetime
    assignee_name: str | None = None


# ---------- Calibration ----------
class CalibrationRunIn(BaseModel):
    run_date: str
    type: str = "extrinsic"
    label: str = ""
    reprojection_rmse_px: float | None = None
    volumetric_scale_rmse_mm: float | None = None
    matched_observations: int | None = None
    num_cameras: int | None = None
    per_camera_rmse: dict | None = None
    notes: str = ""


class CalibrationRunOut(CalibrationRunIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    status: str = "unknown"  # computed: pass / warn / fail / unknown


# ---------- Equipment ----------
class EquipmentIn(BaseModel):
    name: str
    category: str = ""
    quantity: int = 1
    unit_cost: float | None = None
    currency: str = "USD"
    supplier: str = ""
    url: str = ""
    status: str = "owned"
    notes: str = ""


class EquipmentOut(EquipmentIn):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- 3D Models ----------
class Model3DIn(BaseModel):
    name: str
    purpose: str = ""
    quantity: int = 1
    material: str = "PLA"
    file_link: str = ""
    print_status: str = "not_started"
    printed_by: str = ""
    notes: str = ""


class Model3DOut(Model3DIn):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- Scripts ----------
class ScriptInfo(BaseModel):
    id: str
    name: str
    description: str
    category: str = ""
    command: str = ""
    gui: bool = False
    long_running: bool = False
    last_status: str | None = None
    last_run_id: int | None = None


class ScriptRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    script_id: str
    script_name: str
    command: str
    status: str
    return_code: int | None = None
    started_by: str = ""
    started_at: datetime
    finished_at: datetime | None = None


class ScriptRunDetail(ScriptRunOut):
    output: str = ""


class ScriptRunRequest(BaseModel):
    args: str = ""            # extra CLI args appended to the base command
    started_by: str = ""


# ---------- Settings ----------
class SettingIn(BaseModel):
    value: str
