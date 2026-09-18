"""
Database layer for the Employee Burnout Prediction System.
Uses SQLite for lightweight, zero-config storage of employees, predictions, alerts, and user accounts.
"""

import os
import json
import sqlite3
from datetime import datetime

# Database file path (project root)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "burnout.db")


def get_connection():
    """Create a new database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize the database schema and seed default users if empty."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            gender TEXT CHECK(gender IN ('Male', 'Female')),
            company_type TEXT CHECK(company_type IN ('Service', 'Product')),
            wfh_available TEXT CHECK(wfh_available IN ('Yes', 'No')),
            designation REAL,
            resource_allocation REAL,
            mental_fatigue_score REAL,
            favourite_activities TEXT DEFAULT 'listening to music',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
            burn_probability REAL NOT NULL,
            risk_level TEXT CHECK(risk_level IN ('Low', 'Medium', 'High')),
            input_features TEXT,
            model_version TEXT,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
            hours_to_burnout REAL,
            activity TEXT,
            message TEXT,
            break_duration_mins INTEGER DEFAULT 15,
            status TEXT DEFAULT 'Dispatched',
            dispatched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT CHECK(role IN ('HR Manager', 'Administrator', 'Machine Learning Engineer')) NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_employee ON predictions(employee_id);
        CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(predicted_at);
        CREATE INDEX IF NOT EXISTS idx_alerts_employee ON alerts(employee_id);
    """)

    # Schema migration checks for existing databases
    cursor.execute("PRAGMA table_info(employees)")
    emp_cols = [row["name"] for row in cursor.fetchall()]
    if "email" not in emp_cols:
        cursor.execute("ALTER TABLE employees ADD COLUMN email TEXT")
    if "favourite_activities" not in emp_cols:
        cursor.execute("ALTER TABLE employees ADD COLUMN favourite_activities TEXT DEFAULT 'listening to music'")

    cursor.execute("PRAGMA table_info(predictions)")
    pred_cols = [row["name"] for row in cursor.fetchall()]
    if "input_features" not in pred_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN input_features TEXT")
    if "model_version" not in pred_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT")

    # Seed default system users if users table is empty
    user_count = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if user_count == 0:
        from auth import hash_password
        default_users = [
            ("admin", hash_password("admin123"), "Administrator", "System Admin", "admin@company.com"),
            ("hrmanager", hash_password("hr123"), "HR Manager", "HR Lead", "hr@company.com"),
            ("mlengineer", hash_password("ml123"), "Machine Learning Engineer", "ML Team", "ml@company.com")
        ]
        cursor.executemany("""
            INSERT INTO users (username, password_hash, role, name, email)
            VALUES (?, ?, ?, ?, ?)
        """, default_users)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


# -------------------------------------------------------------------
# Employee CRUD operations
# -------------------------------------------------------------------

def create_employee(data):
    """Insert a new employee. Returns the created employee dict."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO employees (employee_id, name, email, gender, company_type,
                                   wfh_available, designation, resource_allocation,
                                   mental_fatigue_score, favourite_activities)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["employee_id"],
            data["name"],
            data.get("email", f"{data['employee_id'].lower()}@company.com"),
            data["gender"],
            data["company_type"],
            data["wfh_available"],
            data.get("designation", 0),
            data.get("resource_allocation", 0),
            data.get("mental_fatigue_score", 0),
            data.get("favourite_activities", "listening to music"),
        ))
        conn.commit()
        return get_employee(cursor.lastrowid)
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise ValueError(f"Employee with ID '{data['employee_id']}' already exists.") from e
    finally:
        conn.close()


def get_employee(employee_pk):
    """Get a single employee by primary key (integer id)."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_pk,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_employees(page=1, per_page=50, search=None):
    """Get paginated list of employees with optional search."""
    conn = get_connection()
    offset = (page - 1) * per_page

    if search:
        search_term = f"%{search}%"
        count = conn.execute(
            "SELECT COUNT(*) FROM employees WHERE name LIKE ? OR employee_id LIKE ?",
            (search_term, search_term)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM employees WHERE name LIKE ? OR employee_id LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (search_term, search_term, per_page, offset)
        ).fetchall()
    else:
        count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM employees ORDER BY id DESC LIMIT ? OFFSET ?",
            (per_page, offset)
        ).fetchall()

    conn.close()
    return {
        "employees": [dict(r) for r in rows],
        "total": count,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (count + per_page - 1) // per_page),
    }


def update_employee(employee_pk, data):
    """Update employee fields. Returns updated employee dict or None."""
    conn = get_connection()
    fields = []
    values = []
    allowed = ["name", "email", "gender", "company_type", "wfh_available",
               "designation", "resource_allocation", "mental_fatigue_score", "favourite_activities"]

    for key in allowed:
        if key in data:
            fields.append(f"{key} = ?")
            values.append(data[key])

    if not fields:
        conn.close()
        return get_employee(employee_pk)

    fields.append("updated_at = ?")
    values.append(datetime.utcnow().isoformat())
    values.append(employee_pk)

    conn.execute(
        f"UPDATE employees SET {', '.join(fields)} WHERE id = ?",
        values
    )
    conn.commit()
    result = get_employee(employee_pk)
    conn.close()
    return result


def delete_employee(employee_pk):
    """Delete an employee by primary key. Returns True if deleted."""
    conn = get_connection()
    cursor = conn.execute("DELETE FROM employees WHERE id = ?", (employee_pk,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# -------------------------------------------------------------------
# Prediction operations
# -------------------------------------------------------------------

def save_prediction(employee_pk, burn_probability, risk_level, input_features=None, model_version=None):
    """Save a prediction result."""
    conn = get_connection()
    cursor = conn.cursor()
    
    features_json = json.dumps(input_features) if isinstance(input_features, dict) else input_features

    cursor.execute("""
        INSERT INTO predictions (employee_id, burn_probability, risk_level, input_features, model_version)
        VALUES (?, ?, ?, ?, ?)
    """, (employee_pk, burn_probability, risk_level, features_json, model_version))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id


def get_predictions(page=1, per_page=20):
    """Get paginated prediction history with employee names."""
    conn = get_connection()
    offset = (page - 1) * per_page

    count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rows = conn.execute("""
        SELECT p.id, p.burn_probability, p.risk_level, p.input_features, p.model_version, p.predicted_at,
               e.name AS employee_name, e.employee_id AS emp_code, e.id AS emp_pk
        FROM predictions p
        JOIN employees e ON p.employee_id = e.id
        ORDER BY p.predicted_at DESC
        LIMIT ? OFFSET ?
    """, (per_page, offset)).fetchall()

    conn.close()
    return {
        "predictions": [dict(r) for r in rows],
        "total": count,
        "page": page,
        "per_page": per_page,
    }


def get_dashboard_stats():
    """Aggregate statistics for the dashboard overview."""
    conn = get_connection()

    total_employees = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    # Latest prediction per employee
    risk_data = conn.execute("""
        SELECT risk_level, COUNT(*) as cnt
        FROM (
            SELECT p.risk_level,
                   ROW_NUMBER() OVER (PARTITION BY p.employee_id ORDER BY p.predicted_at DESC) as rn
            FROM predictions p
        ) sub
        WHERE rn = 1
        GROUP BY risk_level
    """).fetchall()

    risk_distribution = {row["risk_level"]: row["cnt"] for row in risk_data}

    avg_burn = conn.execute("""
        SELECT AVG(burn_probability) FROM (
            SELECT burn_probability,
                   ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY predicted_at DESC) as rn
            FROM predictions
        ) WHERE rn = 1
    """).fetchone()[0]

    # Recent predictions (last 10)
    recent = conn.execute("""
        SELECT p.id, p.burn_probability, p.risk_level, p.predicted_at,
               e.name AS employee_name, e.employee_id AS emp_code
        FROM predictions p
        JOIN employees e ON p.employee_id = e.id
        ORDER BY p.predicted_at DESC
        LIMIT 10
    """).fetchall()

    conn.close()

    return {
        "total_employees": total_employees,
        "total_predictions": total_predictions,
        "high_risk_count": risk_distribution.get("High", 0),
        "medium_risk_count": risk_distribution.get("Medium", 0),
        "low_risk_count": risk_distribution.get("Low", 0),
        "avg_burn_probability": round(avg_burn, 4) if avg_burn else 0.0,
        "risk_distribution": risk_distribution,
        "recent_predictions": [dict(r) for r in recent],
    }


# -------------------------------------------------------------------
# Alert operations
# -------------------------------------------------------------------

def save_burnout_alert(employee_pk, hours_to_burnout, activity, message, break_duration_mins=15):
    """Save a dispatched burnout micro-break alert."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts (employee_id, hours_to_burnout, activity, message, break_duration_mins)
        VALUES (?, ?, ?, ?, ?)
    """, (employee_pk, hours_to_burnout, activity, message, break_duration_mins))
    conn.commit()
    alert_id = cursor.lastrowid
    conn.close()
    return alert_id


def get_burnout_alerts(limit=20):
    """Get recent burnout micro-break alerts."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.id, a.hours_to_burnout, a.activity, a.message, a.break_duration_mins,
               a.status, a.dispatched_at, e.name AS employee_name, e.employee_id AS emp_code
        FROM alerts a
        JOIN employees e ON a.employee_id = e.id
        ORDER BY a.dispatched_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------------
# User Account CRUD operations
# -------------------------------------------------------------------

def get_user_by_username(username):
    """Fetch user by username."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    """Fetch user by user_id."""
    conn = get_connection()
    row = conn.execute("SELECT id, username, role, name, email, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users():
    """List all system user accounts (excluding hashes)."""
    conn = get_connection()
    rows = conn.execute("SELECT id, username, role, name, email, created_at FROM users ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_user(username, password_hash, role, name, email=""):
    """Create a new system user account."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, role, name, email)
            VALUES (?, ?, ?, ?, ?)
        """, (username, password_hash, role, name, email))
        conn.commit()
        return get_user_by_id(cursor.lastrowid)
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise ValueError(f"User with username '{username}' already exists.") from e
    finally:
        conn.close()


def delete_user(user_id):
    """Delete a user account by primary key."""
    conn = get_connection()
    cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted
