"""
Streamlit Analytics & HR Management Dashboard for Employee Burnout Prediction System.
Run with: streamlit run streamlit_app.py
"""

import os
import sys
import pandas as pd
import streamlit as st

# Add src folder to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from database import (
    init_db, get_dashboard_stats, get_all_employees,
    get_employee, save_burnout_alert, get_all_users, create_user
)
from api.predict import predict_single, get_model_info
from drift import detect_feature_drift

st.set_page_config(
    page_title="Burn-out Barrior — Streamlit Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Lavender & Black Theme Styling for Streamlit
st.markdown("""
<style>
    .stApp {
        background-color: #09090d;
        color: #f5f3ff;
    }
    [data-testid="stSidebar"] {
        background-color: #13111c;
        border-right: 1px solid #2e2744;
    }
    [data-testid="stHeader"] {
        background-color: rgba(9, 9, 13, 0.8);
    }
    .stButton>button {
        background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #c084fc 0%, #8b5cf6 100%);
        box-shadow: 0 4px 15px rgba(167, 139, 250, 0.3);
    }
    [data-testid="stMetric"] {
        background-color: #13111c;
        border: 1px solid #2e2744;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #2e2744;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #13111c;
        border-radius: 8px 8px 0 0;
        border: 1px solid #2e2744;
        color: #c4b5fd;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7c3aed !important;
        color: white !important;
        border-color: #a78bfa !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize DB on load
init_db()

# Sidebar Brand & Role Switcher
st.sidebar.title("🔥 Burn-out Barrior")
st.sidebar.caption("AI-Powered Employee Burnout Prediction System")
st.sidebar.markdown("---")

role = st.sidebar.selectbox(
    "Select Operating Role:",
    ["HR Manager", "Administrator", "Machine Learning Engineer"],
    index=0
)

st.sidebar.info(f"Current Persona: **{role}**")
st.sidebar.markdown("---")

# Navigation Tabs based on Role
if role == "HR Manager":
    tabs = st.tabs(["📊 HR Overview", "🎯 Burnout Prediction", "👥 Employee Management", "📑 Report Export"])
    
    # Tab 1: HR Overview
    with tabs[0]:
        st.header("HR Burnout Analytics Overview")
        stats = get_dashboard_stats()
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Employees", stats["total_employees"])
        c2.metric("High Risk 🚨", stats["high_risk_count"], delta_color="inverse")
        c3.metric("Medium Risk ⚡", stats.get("medium_risk_count", 0))
        c4.metric("Low Risk ✅", stats["low_risk_count"])
        c5.metric("Avg Burn Probability", f"{stats['avg_burn_probability'] * 100:.1f}%")
        
        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Risk Level Distribution")
            dist = stats.get("risk_distribution", {})
            if dist:
                df_dist = pd.DataFrame(list(dist.items()), columns=["Risk Level", "Employee Count"])
                st.bar_chart(df_dist.set_index("Risk Level"))
            else:
                st.info("No prediction data available yet.")
                
        with col_right:
            st.subheader("Recent Predictions History")
            recent = stats.get("recent_predictions", [])
            if recent:
                df_recent = pd.DataFrame(recent)[["employee_name", "emp_code", "burn_probability", "risk_level", "predicted_at"]]
                df_recent["burn_probability"] = (df_recent["burn_probability"] * 100).round(1).astype(str) + "%"
                st.dataframe(df_recent, use_container_width=True)
            else:
                st.info("No recent predictions.")

    # Tab 2: Burnout Prediction
    with tabs[1]:
        st.header("Real-Time Burnout Prediction & Mind-Cooling Micro-Break Dispatch")
        
        col_form, col_res = st.columns([1, 1])
        
        with col_form:
            st.subheader("Employee Data Input")
            emp_name = st.text_input("Employee Name", "Krishni")
            fav_act = st.text_input("Favourite Free Time Activity", "listening to music")
            gender = st.selectbox("Gender", ["Female", "Male"])
            company = st.selectbox("Company Type", ["Service", "Product"])
            wfh = st.selectbox("WFH Available", ["Yes", "No"])
            designation = st.slider("Designation (0–5)", 0.0, 5.0, 3.0, 0.5)
            resource = st.slider("Resource Allocation (1–10)", 1.0, 10.0, 8.5, 0.1)
            fatigue = st.slider("Mental Fatigue Score (0–10)", 0.0, 10.0, 7.8, 0.1)
            
            run_btn = st.button("🎯 Run Burnout Prediction", type="primary", use_container_width=True)
            
        with col_res:
            st.subheader("Prediction Output")
            if run_btn:
                payload = {
                    "name": emp_name,
                    "favourite_activities": fav_act,
                    "gender": gender,
                    "company_type": company,
                    "wfh_available": wfh,
                    "designation": designation,
                    "resource_allocation": resource,
                    "mental_fatigue_score": fatigue
                }
                res = predict_single(payload)
                prob = res["burn_probability"] * 100
                risk = res["risk_level"]
                alert = res["burnout_alert"]
                
                st.progress(int(prob))
                
                if risk == "High":
                    st.error(f"⚠️ **High Risk Detected: {prob:.1f}% Burn Probability**")
                elif risk == "Medium":
                    st.warning(f"⚡ **Medium Risk Detected: {prob:.1f}% Burn Probability**")
                else:
                    st.success(f"✅ **Low Risk Detected: {prob:.1f}% Burn Probability**")
                    
                st.markdown("---")
                st.subheader("🚨 Predictive Burnout Alert")
                st.info(f"⏳ **Burnout Predicted in {alert['hours_to_burnout']} hours**")
                st.success(f"🎵 **Personalized Mind-Cooling Recommendation:**\n\n\"{alert['message']}\"")
                st.caption(f"⏰ **Working Hours Protection:** {alert['working_hours_protection']}")
                
                if st.button("🔔 Dispatch Micro-Break Alert to Employee", use_container_width=True):
                    st.toast(f"Micro-break alert dispatched to {emp_name}!", icon="✓")
                    st.balloons()

    # Tab 3: Employee Directory
    with tabs[2]:
        st.header("Employee Directory")
        emps = get_all_employees(page=1, per_page=1000)
        df_emps = pd.DataFrame(emps.get("employees", []))
        if not df_emps.empty:
            st.dataframe(df_emps[["employee_id", "name", "email", "gender", "company_type", "wfh_available", "favourite_activities", "mental_fatigue_score"]], use_container_width=True)
        else:
            st.info("No employees registered yet.")

    # Tab 4: Report Export
    with tabs[3]:
        st.header("HR Burnout Analytics Report Export")
        st.write("Generate and download comprehensive employee burnout CSV reports for executive reviews.")
        emps = get_all_employees(page=1, per_page=1000).get("employees", [])
        if emps:
            df_export = pd.DataFrame(emps)
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download HR Burnout Report (CSV)",
                data=csv_data,
                file_name="hr_burnout_analytics_report.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.warning("No data to export.")

elif role == "Administrator":
    st.header("🛠️ Administrator Portal & System Configuration")
    
    st.subheader("User Account Management")
    users = get_all_users()
    if users:
        df_users = pd.DataFrame(users)
        st.dataframe(df_users[["id", "username", "role", "name", "email", "created_at"]], use_container_width=True)
        
    st.markdown("---")
    st.subheader("Add New System User")
    with st.form("add_user_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("System Role", ["HR Manager", "Administrator", "Machine Learning Engineer"])
        new_name = st.text_input("Full Name")
        new_email = st.text_input("Email")
        submit_user = st.form_submit_button("➕ Create User")
        
        if submit_user:
            if new_username and new_password and new_name:
                from auth import hash_password
                try:
                    create_user(new_username, hash_password(new_password), new_role, new_name, new_email)
                    st.success(f"User '{new_username}' created successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(str(e))
            else:
                st.error("Please fill in all required fields.")

elif role == "Machine Learning Engineer":
    st.header("🧠 Machine Learning Engineer Portal")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("Active Model Architecture")
        info = get_model_info()
        st.json({
            "model_name": info.get("name"),
            "input_shape": info.get("input_shape"),
            "total_params": info.get("total_params"),
            "model_version": info.get("model_version"),
            "metrics": info.get("metrics")
        })
        
        st.markdown("---")
        st.subheader("Model Retraining Engine")
        epochs = st.number_input("Training Epochs", 5, 50, 15)
        batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
        lr = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)
        
        if st.button("🚀 Trigger Model Retraining", type="primary", use_container_width=True):
            with st.spinner("Retraining CNN-LSTM model..."):
                from train_model import retrain_model
                res = retrain_model(epochs=epochs, batch_size=batch_size, learning_rate=lr)
                st.success("Model retrained successfully!")
                st.json(res)
                
    with col_m2:
        st.subheader("Model Drift Monitoring")
        drift = detect_feature_drift()
        st.metric("Overall Drift Index", drift["overall_drift_index"], delta="Stable" if drift["status"] == "healthy" else "Warning")
        st.dataframe(pd.DataFrame(drift["feature_metrics"]), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Training Evaluation Curves")
        plot_path = os.path.join(PROJECT_DIR, "plots", "evaluation_results.png")
        if os.path.exists(plot_path):
            st.image(plot_path, caption="ROC Curve and Confusion Matrix", use_column_width=True)
