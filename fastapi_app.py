"""
FastAPI application for the Employee Burnout Prediction System.
Provides a modern asynchronous REST API backend alongside Flask.
Run with: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Query, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from database import (
    init_db, get_employee, get_all_employees, create_employee,
    update_employee, delete_employee, save_prediction, get_predictions,
    get_dashboard_stats, save_burnout_alert, get_burnout_alerts,
    get_user_by_username, get_user_by_id, get_all_users, create_user, delete_user
)
from api.predict import (
    predict_single, predict_batch, get_model_info, MODEL_VERSION,
    analyze_contributing_factors, get_interventions
)
from auth import verify_password, generate_token, decode_token, hash_password
from drift import detect_feature_drift

# Initialize FastAPI app
app = FastAPI(
    title="Employee Burnout Prediction System API",
    description="FastAPI Backend for AI-powered CNN-LSTM Burnout Risk Prediction & HR Analytics",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()

# Pydantic Schemas
class EmployeeCreate(BaseModel):
    employee_id: str
    name: str
    gender: str
    company_type: str
    wfh_available: str
    designation: float = 2.0
    resource_allocation: float = 5.0
    mental_fatigue_score: float = 5.0
    email: Optional[str] = ""
    favourite_activities: Optional[str] = "listening to music"

class PredictRequest(BaseModel):
    employee_id: Optional[int] = None
    gender: Optional[str] = None
    company_type: Optional[str] = None
    wfh_available: Optional[str] = None
    designation: Optional[float] = None
    resource_allocation: Optional[float] = None
    mental_fatigue_score: Optional[float] = None
    name: Optional[str] = "Employee"
    favourite_activities: Optional[str] = "listening to music"

class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str
    name: str
    email: Optional[str] = ""

class RetrainRequest(BaseModel):
    epochs: Optional[int] = 15
    batch_size: Optional[int] = 64
    learning_rate: Optional[float] = 0.001

class AlertDispatchRequest(BaseModel):
    employee_id: Optional[int] = None
    name: Optional[str] = "Employee"
    activity: Optional[str] = "listening to music"
    hours_to_burnout: Optional[float] = 1.0
    message: Optional[str] = None


# Endpoints
@app.get("/")
def read_root():
    return {"message": "Employee Burnout Prediction System FastAPI Online", "docs": "/docs"}

@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = get_user_by_username(req.username.strip())
    if not user or not verify_password(req.password.strip(), user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = generate_token(user["id"], user["username"], user["role"])
    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "name": user["name"],
            "email": user["email"]
        }
    }

@app.get("/api/auth/me")
def get_current_user(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "").strip() if authorization and authorization.startswith("Bearer ") else None
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user_by_id(payload["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": user}

@app.get("/api/employees")
def list_emp(page: int = 1, per_page: int = 50, search: Optional[str] = None):
    return get_all_employees(page=page, per_page=per_page, search=search)

@app.get("/api/employees/{pk}")
def get_emp(pk: int):
    emp = get_employee(pk)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp

@app.post("/api/employees", status_code=201)
def add_emp(emp: EmployeeCreate):
    try:
        return create_employee(emp.dict())
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.put("/api/employees/{pk}")
def edit_emp(pk: int, data: Dict[str, Any]):
    emp = get_employee(pk)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return update_employee(pk, data)

@app.delete("/api/employees/{pk}")
def remove_emp(pk: int):
    if delete_employee(pk):
        return {"message": "Employee deleted successfully"}
    raise HTTPException(status_code=404, detail="Employee not found")

@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.employee_id:
        emp = get_employee(req.employee_id)
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")
        features = {
            "name": emp["name"],
            "favourite_activities": emp.get("favourite_activities", "listening to music"),
            "gender": emp["gender"],
            "company_type": emp["company_type"],
            "wfh_available": emp["wfh_available"],
            "designation": emp["designation"],
            "resource_allocation": emp["resource_allocation"],
            "mental_fatigue_score": emp["mental_fatigue_score"],
        }
        res = predict_single(features)
        save_prediction(emp["id"], res["burn_probability"], res["risk_level"], input_features=features, model_version=MODEL_VERSION)
        res["employee_name"] = emp["name"]
        res["employee_id"] = emp["id"]
        res["favourite_activities"] = emp.get("favourite_activities", "listening to music")
        return res

    data = req.dict(exclude_unset=True)
    required = ["gender", "company_type", "wfh_available", "designation", "resource_allocation", "mental_fatigue_score"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")
    return predict_single(data)

@app.post("/api/predict/send-burnout-alert")
def dispatch_alert(req: AlertDispatchRequest):
    if req.employee_id:
        emp = get_employee(req.employee_id)
        if emp:
            features = {
                "name": emp["name"],
                "favourite_activities": emp.get("favourite_activities", "listening to music"),
                "gender": emp["gender"],
                "company_type": emp["company_type"],
                "wfh_available": emp["wfh_available"],
                "designation": emp["designation"],
                "resource_allocation": emp["resource_allocation"],
                "mental_fatigue_score": emp["mental_fatigue_score"],
            }
            res = predict_single(features)
            alert = res["burnout_alert"]
            alert_id = save_burnout_alert(emp["id"], alert["hours_to_burnout"], alert["activity"], alert["message"], alert["break_duration_mins"])
            return {
                "alert_id": alert_id,
                "message": alert["message"],
                "hours_to_burnout": alert["hours_to_burnout"],
                "activity": alert["activity"],
                "recipient": emp["name"],
                "status": "Dispatched"
            }

    name = req.name or "Employee"
    act = req.activity or "listening to music"
    msg = req.message or f"{name} you go {act} for 15 mins that makes your mind cool"
    return {
        "message": msg,
        "hours_to_burnout": req.hours_to_burnout or 1.0,
        "activity": act,
        "recipient": name,
        "status": "Dispatched"
    }

@app.get("/api/dashboard/stats")
def stats():
    return get_dashboard_stats()

@app.get("/api/model/info")
def model_information():
    return get_model_info()

@app.get("/api/ml/drift")
def drift_info():
    return detect_feature_drift()

@app.post("/api/ml/retrain")
def retrain(req: RetrainRequest):
    try:
        from train_model import retrain_model
        return retrain_model(epochs=req.epochs, batch_size=req.batch_size, learning_rate=req.learning_rate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.get("/api/reports/download")
def download_rep():
    employees_res = get_all_employees(page=1, per_page=1000)
    employees = employees_res.get("employees", [])
    csv_lines = ["ID,Employee Code,Name,Email,Gender,Company Type,WFH,Designation,Resource Allocation,Mental Fatigue Score,Favourite Activities"]
    for e in employees:
        csv_lines.append(f'{e["id"]},"{e["employee_id"]}","{e["name"]}","{e.get("email","")}","{e["gender"]}","{e["company_type"]}","{e["wfh_available"]}",{e["designation"]},{e["resource_allocation"]},{e["mental_fatigue_score"]},"{e.get("favourite_activities","")}"')
    csv_content = "\n".join(csv_lines)
    return Response(content=csv_content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=hr_burnout_report.csv"})
