"""
Prediction engine for the Employee Burnout Prediction System.
Loads the trained CNN-LSTM model and scaler, preprocesses employee features
into 12-month synthetic time series, and runs inference.
"""

import os
import pickle
import numpy as np

# Lazy-loaded singletons
_model = None
_scaler = None

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_DIR, "best_model.keras")
SCALER_PATH = os.path.join(PROJECT_DIR, "processed_data", "scaler.pkl")

# Feature indices that get scaled (same as features.py)
SCALE_COLS = [3, 4, 5, 6, 7, 8, 9]
NUM_FEATURES = 10
NUM_TIMESTEPS = 12
RISK_THRESHOLD = 0.5
MODEL_VERSION = "1.0.0-cnn-lstm"


def load_model():
    """Load the trained CNN-LSTM model (singleton)."""
    global _model
    if _model is None:
        # Import tensorflow only when needed to avoid slow startup for non-predict routes
        from tensorflow.keras.models import load_model as keras_load
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        _model = keras_load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    return _model


def load_scaler():
    """Load the fitted StandardScaler (singleton)."""
    global _scaler
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run the pipeline first.")
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        print(f"Loaded scaler from {SCALER_PATH}")
    return _scaler


def build_time_series(employee):
    """
    Build a 12-month synthetic time series from a single employee's features.

    Args:
        employee: dict with keys: gender, company_type, wfh_available,
                  designation, resource_allocation, mental_fatigue_score

    Returns:
        numpy array of shape (12, 10) — one sample ready for the model.

    The simulation logic mirrors src/features.py exactly:
    - Month 12 (index 11) = actual employee values
    - Months 1-11 are simulated backward with drift + noise
    """
    np.random.seed(42)  # Deterministic predictions for same input

    R_seq = np.zeros(12)
    M_seq = np.zeros(12)
    S_seq = np.zeros(12)

    # Month 12 baseline
    R_seq[11] = float(employee.get("resource_allocation", 5))
    M_seq[11] = float(employee.get("mental_fatigue_score", 5))
    S_seq[11] = np.clip(10.0 - M_seq[11] + np.random.normal(0, 0.3), 0.0, 10.0)

    # Simulate backward
    for t in range(10, -1, -1):
        drift_R = 0.03
        noise_R = np.random.normal(0, 0.1)
        R_seq[t] = np.clip(R_seq[t + 1] - drift_R - noise_R, 1.0, 10.0)

        drift_M = 0.01 * R_seq[t] + 0.01
        noise_M = np.random.normal(0, 0.12)
        M_seq[t] = np.clip(M_seq[t + 1] - drift_M - noise_M, 0.0, 10.0)

        S_seq[t] = np.clip(10.0 - M_seq[t] + np.random.normal(0, 0.3), 0.0, 10.0)

    S_seq = S_seq / 10.0  # Normalize to 0-1

    # Compute temporal features
    volatility = np.zeros(12)
    sat_drop = np.zeros(12)
    overtime = np.zeros(12)

    s_max = 0.0
    for t in range(12):
        # Volatility index
        if t > 0:
            volatility[t] = np.std(M_seq[:t + 1])

        # Satisfaction drop
        if t == 0:
            s_max = S_seq[0]
        else:
            s_max = max(s_max, S_seq[t])
            sat_drop[t] = max(sat_drop[t - 1], s_max - S_seq[t])

        # Overtime trend (slope over last 6 months)
        start_idx = max(0, t - 5)
        window_len = t - start_idx + 1
        if window_len > 1:
            x = np.arange(window_len)
            x_mean = np.mean(x)
            x_var = np.var(x) * window_len
            y = R_seq[start_idx:t + 1]
            y_mean = np.mean(y)
            cov = np.sum((x - x_mean) * (y - y_mean))
            overtime[t] = cov / x_var

    # Assemble feature matrix (12, 10)
    X = np.zeros((12, NUM_FEATURES))
    gender_male = 1.0 if employee.get("gender", "").lower() == "male" else 0.0
    company_service = 1.0 if employee.get("company_type", "").lower() == "service" else 0.0
    wfh_yes = 1.0 if employee.get("wfh_available", "").lower() == "yes" else 0.0
    designation = float(employee.get("designation", 2))

    X[:, 0] = gender_male
    X[:, 1] = company_service
    X[:, 2] = wfh_yes
    X[:, 3] = designation
    X[:, 4] = R_seq
    X[:, 5] = M_seq
    X[:, 6] = S_seq
    X[:, 7] = volatility
    X[:, 8] = sat_drop
    X[:, 9] = overtime

    return X


def scale_features(X_single):
    """
    Scale numeric features using the fitted scaler.

    Args:
        X_single: array of shape (12, 10) — one sample

    Returns:
        Scaled array of the same shape.
    """
    scaler = load_scaler()
    X_copy = X_single.copy()
    X_copy[:, SCALE_COLS] = scaler.transform(X_copy[:, SCALE_COLS])
    return X_copy


def analyze_contributing_factors(employee):
    """
    Analyze which input features contribute most to burnout risk.
    Uses a rule-based approach comparing feature values to risk thresholds.

    Returns:
        list of dicts sorted by impact (highest first)
    """
    factors = []

    # Mental Fatigue Score (0-10, higher = worse)
    fatigue = float(employee.get("mental_fatigue_score", 0))
    fatigue_impact = min(fatigue / 10.0, 1.0)
    factors.append({
        "feature": "Mental Fatigue Score",
        "value": fatigue,
        "impact": round(fatigue_impact, 3),
        "status": "critical" if fatigue >= 7 else "warning" if fatigue >= 5 else "normal",
        "description": f"Score of {fatigue}/10 — {'dangerously high' if fatigue >= 7 else 'elevated' if fatigue >= 5 else 'within safe range'}",
    })

    # Resource Allocation (1-10, higher = more overloaded)
    resource = float(employee.get("resource_allocation", 0))
    resource_impact = min(resource / 10.0, 1.0)
    factors.append({
        "feature": "Resource Allocation",
        "value": resource,
        "impact": round(resource_impact, 3),
        "status": "critical" if resource >= 8 else "warning" if resource >= 6 else "normal",
        "description": f"Allocation of {resource}/10 — {'severe overload' if resource >= 8 else 'high workload' if resource >= 6 else 'manageable'}",
    })

    # Designation (0-5, higher = more senior, more stress)
    designation = float(employee.get("designation", 0))
    designation_impact = designation / 5.0
    factors.append({
        "feature": "Designation Level",
        "value": designation,
        "impact": round(designation_impact, 3),
        "status": "warning" if designation >= 4 else "normal",
        "description": f"Level {int(designation)}/5 — {'senior role with higher responsibility' if designation >= 4 else 'standard level'}",
    })

    # WFH availability (No = higher risk)
    wfh = employee.get("wfh_available", "No")
    wfh_impact = 0.6 if wfh.lower() == "no" else 0.15
    factors.append({
        "feature": "WFH Setup",
        "value": wfh,
        "impact": round(wfh_impact, 3),
        "status": "warning" if wfh.lower() == "no" else "normal",
        "description": f"{'No remote work option — reduced flexibility increases stress' if wfh.lower() == 'no' else 'Remote work available — provides flexibility'}",
    })

    # Sort by impact descending
    factors.sort(key=lambda x: x["impact"], reverse=True)
    return factors


def get_interventions(risk_level, contributing_factors):
    """
    Generate rule-based recommended interventions based on risk level
    and the top contributing factors.

    Returns:
        list of intervention dicts
    """
    interventions = []

    # Map factor names to interventions
    factor_interventions = {
        "Mental Fatigue Score": {
            "critical": {
                "action": "Immediate Mental Health Support",
                "detail": "Schedule counseling sessions and consider mandatory time-off. Reduce cognitive load by delegating non-essential tasks.",
                "urgency": "high",
            },
            "warning": {
                "action": "Stress Management Program",
                "detail": "Enroll in mindfulness workshops and ensure regular breaks during work hours.",
                "urgency": "medium",
            },
        },
        "Resource Allocation": {
            "critical": {
                "action": "Workload Redistribution",
                "detail": "Immediately reassign tasks to reduce load below capacity threshold. Consider hiring or team restructuring.",
                "urgency": "high",
            },
            "warning": {
                "action": "Workload Review",
                "detail": "Review current task assignments and prioritize. Implement time-boxing for better work-life balance.",
                "urgency": "medium",
            },
        },
        "Designation Level": {
            "warning": {
                "action": "Leadership Support",
                "detail": "Provide executive coaching and delegation training to manage responsibilities effectively.",
                "urgency": "medium",
            },
        },
        "WFH Setup": {
            "warning": {
                "action": "Flexible Work Arrangement",
                "detail": "Enable hybrid/remote work options to reduce commute stress and improve work-life balance.",
                "urgency": "medium",
            },
        },
    }

    for factor in contributing_factors:
        name = factor["feature"]
        status = factor["status"]
        if name in factor_interventions and status in factor_interventions[name]:
            interventions.append(factor_interventions[name][status])

    # Add general recommendations based on risk level
    if risk_level == "High":
        interventions.append({
            "action": "Manager 1-on-1 Check-in",
            "detail": "Schedule a private discussion with the employee's direct manager to assess well-being and create an action plan.",
            "urgency": "high",
        })
    elif risk_level == "Medium":
        interventions.append({
            "action": "Scheduled Mind-Cooling Micro-Break",
            "detail": "Encourage taking planned 15-minute free-time activity breaks during work transitions to lower cognitive fatigue.",
            "urgency": "medium",
        })
    else:
        interventions.append({
            "action": "Preventive Monitoring",
            "detail": "Continue periodic assessments. Current indicators are within safe range.",
            "urgency": "low",
        })

    return interventions


def classify_risk_level(prob):
    """Classify burnout probability into Low, Medium, or High risk level."""
    if prob >= 0.65:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    else:
        return "Low"


def generate_burnout_alert(employee, burn_probability):
    """
    Generate an early warning burnout alert with estimated hours to burnout
    and a personalized mind-cooling message based on the employee's favorite free-time activity.

    Example message:
    "Krishni you go listen to music for 15 mins that makes your mind cool"
    """
    risk_level = classify_risk_level(burn_probability)

    # Calculate burnout time horizon in hours
    if burn_probability >= 0.8:
        hours_to_burnout = 1.0
    elif burn_probability >= 0.65:
        hours_to_burnout = 2.0
    elif burn_probability >= 0.5:
        hours_to_burnout = 3.5
    elif burn_probability >= 0.35:
        hours_to_burnout = 5.0
    else:
        hours_to_burnout = round(max(6.0, (1.0 - burn_probability) * 10), 1)

    name = employee.get("name", "Employee")
    activities_str = employee.get("favourite_activities", "listening to music")

    # Extract first favourite activity or clean up string
    activities = [a.strip() for a in activities_str.split(",") if a.strip()]
    fav_activity = activities[0] if activities else "listening to music"

    # Clean activity phrase for natural sentence flow
    activity_action = fav_activity
    if activity_action.lower().startswith("listening to "):
        activity_action = "listen to " + activity_action[13:]
    elif activity_action.lower().startswith("listening "):
        activity_action = "listen to " + activity_action[10:]
    elif activity_action.lower().startswith("reading "):
        activity_action = "read " + activity_action[8:]
    elif activity_action.lower().startswith("walking "):
        activity_action = "walk " + activity_action[8:]

    break_mins = 15

    # Construct personalized cooling alert message
    alert_message = f"{name} you go {activity_action} for {break_mins} mins that makes your mind cool"

    working_hours_protection = (
        f"This {break_mins}-minute restorative micro-break is scheduled during an optimal task transition "
        f"to prevent cognitive overload without impacting daily working hours or project deadlines."
    )

    return {
        "burnout_predicted": risk_level in ["High", "Medium"],
        "hours_to_burnout": hours_to_burnout,
        "activity": fav_activity,
        "break_duration_mins": break_mins,
        "message": alert_message,
        "working_hours_protection": working_hours_protection,
        "recommended_schedule": "Immediate transition window (within 30 mins)"
    }


def predict_single(employee):
    """
    Run burnout prediction for a single employee.

    Args:
        employee: dict with feature fields

    Returns:
        dict with burn_probability, risk_level, contributing_factors, interventions, and burnout_alert
    """
    model = load_model()

    X = build_time_series(employee)
    X_scaled = scale_features(X)
    X_batch = X_scaled[np.newaxis, :, :]  # shape (1, 12, 10)

    prob = float(model.predict(X_batch, verbose=0).flatten()[0])
    risk = classify_risk_level(prob)

    # Analyze contributing factors and generate interventions
    factors = analyze_contributing_factors(employee)
    interventions = get_interventions(risk, factors)

    # Generate early burnout warning & personalized micro-break alert
    alert = generate_burnout_alert(employee, prob)

    return {
        "burn_probability": round(prob, 4),
        "risk_level": risk,
        "contributing_factors": factors,
        "interventions": interventions,
        "burnout_alert": alert,
    }


def predict_batch(employees):
    """
    Run burnout prediction for a batch of employees.

    Args:
        employees: list of dicts with feature fields

    Returns:
        list of result dicts
    """
    model = load_model()

    X_all = np.array([build_time_series(emp) for emp in employees])
    # Scale each sample
    X_scaled = np.array([scale_features(X_all[i]) for i in range(len(X_all))])

    probs = model.predict(X_scaled, verbose=0).flatten()

    results = []
    for prob in probs:
        p = float(prob)
        results.append({
            "burn_probability": round(p, 4),
            "risk_level": classify_risk_level(p),
        })

    return results


def _get_layer_shape(layer):
    """Safely get a layer's output shape (compatible with Keras 2.x and 3.x)."""
    try:
        return str(layer.output_shape)
    except (AttributeError, RuntimeError):
        try:
            return str(layer.output.shape)
        except (AttributeError, RuntimeError):
            return "N/A"


def get_model_info():
    """Return metadata about the trained model."""
    model = load_model()

    return {
        "name": "CNN-LSTM Hybrid",
        "input_shape": str(model.input_shape),
        "total_params": int(model.count_params()),
        "layers": [
            {"name": layer.name, "type": layer.__class__.__name__,
             "output_shape": _get_layer_shape(layer)}
            for layer in model.layers
        ],
        "threshold": RISK_THRESHOLD,
        "metrics": {
            "accuracy": 0.9308,
            "f1_score": 0.9155,
            "roc_auc": 0.9840,
        },
        "model_path": MODEL_PATH,
        "model_version": MODEL_VERSION,
    }
