"""
REST API routes for the Employee Burnout Prediction System.
Provides endpoints for employee CRUD, burnout predictions, and dashboard statistics.
"""

import os
import sys
from flask import Blueprint, request, jsonify

# Add src to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import (
    create_employee, get_employee, get_all_employees,
    update_employee, delete_employee,
    save_prediction, get_predictions, get_dashboard_stats,
    save_burnout_alert, get_burnout_alerts,
)
from api.predict import (
    predict_single, predict_batch, get_model_info, MODEL_VERSION,
    analyze_contributing_factors, get_interventions, generate_burnout_alert
)

api = Blueprint("api", __name__, url_prefix="/api")


# -------------------------------------------------------------------
# Employee endpoints
# -------------------------------------------------------------------

@api.route("/employees", methods=["GET"])
def list_employees():
    """List all employees (paginated, searchable)."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    search = request.args.get("search", None)
    result = get_all_employees(page=page, per_page=per_page, search=search)
    return jsonify(result)


@api.route("/employees/<int:employee_pk>", methods=["GET"])
def get_single_employee(employee_pk):
    """Get an employee by primary key."""
    emp = get_employee(employee_pk)
    if emp is None:
        return jsonify({"error": "Employee not found"}), 404
    return jsonify(emp)


@api.route("/employees", methods=["POST"])
def add_employee():
    """Add a new employee."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    required = ["employee_id", "name", "gender", "company_type", "wfh_available"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        emp = create_employee(data)
        return jsonify(emp), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 409


@api.route("/employees/<int:employee_pk>", methods=["PUT"])
def edit_employee(employee_pk):
    """Update an employee's details."""
    emp = get_employee(employee_pk)
    if emp is None:
        return jsonify({"error": "Employee not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    updated = update_employee(employee_pk, data)
    return jsonify(updated)


@api.route("/employees/<int:employee_pk>", methods=["DELETE"])
def remove_employee(employee_pk):
    """Delete an employee."""
    if delete_employee(employee_pk):
        return jsonify({"message": "Employee deleted successfully"}), 200
    return jsonify({"error": "Employee not found"}), 404


@api.route("/employees/<int:employee_pk>/send-relief", methods=["POST"])
def send_relief_email(employee_pk):
    """Simulate sending a stress-relief email to an employee."""
    emp = get_employee(employee_pk)
    if emp is None:
        return jsonify({"error": "Employee not found"}), 404

    email = emp.get("email")
    if not email:
        email = f"{emp['employee_id'].lower()}@company.com"

    features = {
        "gender": emp["gender"],
        "company_type": emp["company_type"],
        "wfh_available": emp["wfh_available"],
        "designation": emp["designation"],
        "resource_allocation": emp["resource_allocation"],
        "mental_fatigue_score": emp["mental_fatigue_score"],
    }

    # Analyze to get interventions
    factors = analyze_contributing_factors(features)
    interventions = get_interventions("High", factors)

    # Simulate email drafting in console
    print(f"\n{'='*60}")
    print(f"📧 EMAIL DISPATCH SIMULATOR")
    print(f"To: {emp['name']} <{email}>")
    print(f"Subject: Burn-out Barrior Well-being Check-in & Resources")
    print(f"{'='*60}")
    print(f"Hi {emp['name']},\n")
    print("We noticed you've been working incredibly hard lately. Your well-being is our top priority.")
    print("Based on recent indicators, we've put together some resources that might help relieve stress:\n")

    for idx, intervention in enumerate(interventions, 1):
        print(f"{idx}. {intervention['action'].upper()}: {intervention['detail']}")

    print("\nPlease remember it's okay to step back. Consider taking some time off.")
    print("Reach out to HR or reply to this email if you need immediate support.")
    print("\nTake care,\nThe Burn-out Barrior Wellbeing Team")
    print(f"{'='*60}\n")

    return jsonify({"message": f"Supportive resources successfully sent to {email}"}), 200


@api.route("/predict/send-burnout-alert", methods=["POST"])
def dispatch_general_burnout_alert():
    """
    Dispatch early burnout warning micro-break message for either a registered employee or direct payload.
    """
    req_data = request.get_json() or {}
    employee_pk = req_data.get("employee_id")

    if employee_pk and isinstance(employee_pk, int):
        emp = get_employee(employee_pk)
        if emp:
            email = emp.get("email") or f"{emp['employee_id'].lower()}@company.com"
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
            pred_res = predict_single(features)
            alert = pred_res["burnout_alert"]
            alert_id = save_burnout_alert(
                employee_pk=emp["id"],
                hours_to_burnout=alert["hours_to_burnout"],
                activity=alert["activity"],
                message=alert["message"],
                break_duration_mins=alert["break_duration_mins"]
            )
            print(f"\n{'='*60}\n[ALERT] PREDICTIVE BURNOUT ALERT DISPATCH\nTo: {emp['name']} <{email}>\nMessage: \"{alert['message']}\"\n{'='*60}\n")
            return jsonify({
                "alert_id": alert_id,
                "message": alert["message"],
                "hours_to_burnout": alert["hours_to_burnout"],
                "activity": alert["activity"],
                "recipient": emp["name"],
                "recipient_email": email,
                "working_hours_protection": alert["working_hours_protection"],
                "status": "Dispatched"
            }), 200

    name = req_data.get("name") or "Employee"
    activity = req_data.get("activity") or "listening to music"
    hours_to_burnout = req_data.get("hours_to_burnout") or 1.0
    msg = req_data.get("message") or f"{name} you go {activity} for 15 mins that makes your mind cool"

    print(f"\n{'='*60}\n[ALERT] PREDICTIVE BURNOUT ALERT DISPATCH (Direct)\nTo: {name}\nMessage: \"{msg}\"\n{'='*60}\n")

    return jsonify({
        "message": msg,
        "hours_to_burnout": hours_to_burnout,
        "activity": activity,
        "recipient": name,
        "recipient_email": f"{name.lower().replace(' ', '.')}@company.com",
        "working_hours_protection": "Scheduled during optimal work transition period without affecting daily work hours.",
        "status": "Dispatched"
    }), 200


@api.route("/employees/<int:employee_pk>/send-burnout-alert", methods=["POST"])
def dispatch_burnout_alert(employee_pk):
    """
    Dispatch early burnout warning with personalized favourite free-time activity micro-break message.
    """
    emp = get_employee(employee_pk)
    if emp is None:
        return jsonify({"error": "Employee not found"}), 404

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

    # Run prediction to generate current alert details
    pred_res = predict_single(features)
    alert = pred_res["burnout_alert"]

    # Save to database
    alert_id = save_burnout_alert(
        employee_pk=emp["id"],
        hours_to_burnout=alert["hours_to_burnout"],
        activity=alert["activity"],
        message=alert["message"],
        break_duration_mins=alert["break_duration_mins"]
    )

    # Print log in terminal
    email = emp.get("email") or f"{emp['employee_id'].lower()}@company.com"
    print(f"\n{'='*60}")
    print(f"[ALERT] PREDICTIVE BURNOUT ALERT DISPATCH")
    print(f"To: {emp['name']} <{email}>")
    print(f"Hours to Burnout: {alert['hours_to_burnout']} hours")
    print(f"Message: \"{alert['message']}\"")
    print(f"Working Hours Safeguard: {alert['working_hours_protection']}")
    print(f"{'='*60}\n")

    return jsonify({
        "alert_id": alert_id,
        "message": alert["message"],
        "hours_to_burnout": alert["hours_to_burnout"],
        "activity": alert["activity"],
        "recipient": emp["name"],
        "recipient_email": email,
        "working_hours_protection": alert["working_hours_protection"],
        "status": "Dispatched"
    }), 200


# -------------------------------------------------------------------
# Prediction endpoints
# -------------------------------------------------------------------

@api.route("/predict", methods=["POST"])
def predict_employee():
    """
    Predict burnout risk for a single employee.

    Accepts either:
      - { "employee_id": <int pk> }  → looks up employee from DB
      - { "gender": ..., "company_type": ..., "name": ..., "favourite_activities": ... }  → uses provided features directly
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    # If employee_id (pk) is provided, look up from DB
    if "employee_id" in data and isinstance(data["employee_id"], int):
        emp = get_employee(data["employee_id"])
        if emp is None:
            return jsonify({"error": "Employee not found"}), 404
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
        result = predict_single(features)
        # Save prediction to DB with input features
        save_prediction(emp["id"], result["burn_probability"], result["risk_level"],
                        input_features=features, model_version=MODEL_VERSION)
        result["employee_name"] = emp["name"]
        result["employee_id"] = emp["id"]
        result["favourite_activities"] = emp.get("favourite_activities", "listening to music")
        return jsonify(result)

    # Otherwise, predict from provided features (no DB save)
    required = ["gender", "company_type", "wfh_available", "designation",
                 "resource_allocation", "mental_fatigue_score"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    result = predict_single(data)
    return jsonify(result)


@api.route("/alerts", methods=["GET"])
def list_alerts():
    """Get recent burnout micro-break alerts."""
    limit = request.args.get("limit", 20, type=int)
    alerts = get_burnout_alerts(limit=limit)
    return jsonify({"alerts": alerts, "count": len(alerts)})


@api.route("/predict/batch", methods=["POST"])
def predict_employees_batch():
    """
    Batch predict burnout for multiple employees by their PKs.
    Body: { "employee_ids": [1, 2, 3, ...] }
    """
    data = request.get_json()
    if not data or "employee_ids" not in data:
        return jsonify({"error": "employee_ids array is required"}), 400

    employees = []
    emp_records = []
    for pk in data["employee_ids"]:
        emp = get_employee(pk)
        if emp is None:
            return jsonify({"error": f"Employee with id {pk} not found"}), 404
        emp_records.append(emp)
        employees.append({
            "gender": emp["gender"],
            "company_type": emp["company_type"],
            "wfh_available": emp["wfh_available"],
            "designation": emp["designation"],
            "resource_allocation": emp["resource_allocation"],
            "mental_fatigue_score": emp["mental_fatigue_score"],
        })

    results = predict_batch(employees)

    # Save all predictions and enrich results
    for i, result in enumerate(results):
        emp = emp_records[i]
        emp_features = employees[i]
        save_prediction(emp["id"], result["burn_probability"], result["risk_level"],
                        input_features=emp_features, model_version=MODEL_VERSION)
        result["employee_name"] = emp["name"]
        result["employee_id"] = emp["id"]
        result["emp_code"] = emp["employee_id"]

    return jsonify({"predictions": results, "count": len(results)})


@api.route("/predictions", methods=["GET"])
def list_predictions():
    """Get paginated prediction history."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    result = get_predictions(page=page, per_page=per_page)
    return jsonify(result)


# -------------------------------------------------------------------
# Dashboard & model info
# -------------------------------------------------------------------

@api.route("/dashboard/stats", methods=["GET"])
def dashboard_stats():
    """Aggregate dashboard statistics."""
    stats = get_dashboard_stats()
    return jsonify(stats)


@api.route("/model/info", methods=["GET"])
def model_info():
    """Model metadata and performance metrics."""
    try:
        info = get_model_info()
        return jsonify(info)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------
# Authentication & Role-Based Access Control (RBAC)
# -------------------------------------------------------------------

@api.route("/auth/login", methods=["POST"])
def auth_login():
    """Authenticate user credentials and return access token."""
    from auth import verify_password, generate_token
    from database import get_user_by_username
    
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid username or password"}), 401

    token = generate_token(user["id"], user["username"], user["role"])
    user_info = {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "name": user["name"],
        "email": user["email"]
    }

    return jsonify({"message": "Login successful", "token": token, "user": user_info})


@api.route("/auth/me", methods=["GET"])
def auth_me():
    """Get authenticated user info from token."""
    from auth import decode_token
    from database import get_user_by_id
    
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else request.args.get("token", "")

    payload = decode_token(token) if token else None
    if not payload:
        return jsonify({"error": "Unauthorized / Invalid Token"}), 401

    user = get_user_by_id(payload["user_id"])
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"user": user})


# -------------------------------------------------------------------
# Administrator Endpoints
# -------------------------------------------------------------------

@api.route("/admin/users", methods=["GET"])
def list_users():
    """List system users (Admin only)."""
    from database import get_all_users
    users = get_all_users()
    return jsonify({"users": users, "count": len(users)})


@api.route("/admin/users", methods=["POST"])
def add_user():
    """Add a new user with role (Admin only)."""
    from auth import hash_password
    from database import create_user
    
    data = request.get_json() or {}
    required = ["username", "password", "role", "name"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    role = data["role"]
    valid_roles = ["HR Manager", "Administrator", "Machine Learning Engineer"]
    if role not in valid_roles:
        return jsonify({"error": f"Invalid role. Must be one of: {', '.join(valid_roles)}"}), 400

    try:
        user = create_user(
            username=data["username"].strip(),
            password_hash=hash_password(data["password"]),
            role=role,
            name=data["name"].strip(),
            email=data.get("email", "").strip()
        )
        return jsonify({"message": "User created successfully", "user": user}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 409


@api.route("/admin/users/<int:user_id>", methods=["DELETE"])
def remove_user(user_id):
    """Delete a user (Admin only)."""
    from database import delete_user
    if delete_user(user_id):
        return jsonify({"message": "User deleted successfully"}), 200
    return jsonify({"error": "User not found"}), 404


# -------------------------------------------------------------------
# Machine Learning Engineer Endpoints
# -------------------------------------------------------------------

@api.route("/ml/retrain", methods=["POST"])
def trigger_retrain():
    """Trigger CNN-LSTM model retraining (ML Engineer only)."""
    try:
        from train_model import retrain_model
        data = request.get_json() or {}
        epochs = data.get("epochs", 15)
        batch_size = data.get("batch_size", 64)
        learning_rate = data.get("learning_rate", 0.001)

        result = retrain_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500


@api.route("/ml/drift", methods=["GET"])
def get_drift_metrics():
    """Get model feature drift metrics (ML Engineer only)."""
    from drift import detect_feature_drift
    drift_data = detect_feature_drift()
    return jsonify(drift_data)


# -------------------------------------------------------------------
# HR Manager Report Generation
# -------------------------------------------------------------------

@api.route("/reports/download", methods=["GET"])
def download_report():
    """Generate and download HR burnout analytics report (CSV format)."""
    from database import get_all_employees
    from flask import Response

    fmt = request.args.get("format", "csv").lower()
    employees_res = get_all_employees(page=1, per_page=1000)
    employees = employees_res.get("employees", [])

    if fmt == "json":
        return jsonify({"report_date": os.popen("date /t").read().strip(), "employees": employees})

    # Generate CSV format
    csv_lines = ["ID,Employee Code,Name,Email,Gender,Company Type,WFH,Designation,Resource Allocation,Mental Fatigue Score,Favourite Activities"]
    for e in employees:
        csv_lines.append(f'{e["id"]},"{e["employee_id"]}","{e["name"]}","{e.get("email","")}","{e["gender"]}","{e["company_type"]}","{e["wfh_available"]}",{e["designation"]},{e["resource_allocation"]},{e["mental_fatigue_score"]},"{e.get("favourite_activities","")}"')

    csv_content = "\n".join(csv_lines)
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=hr_burnout_report.csv"}
    )
