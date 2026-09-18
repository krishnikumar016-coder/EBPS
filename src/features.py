import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_features():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_data_path = os.path.join(project_dir, "data", "train_cleaned.csv")
    processed_dir = os.path.join(project_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(cleaned_data_path):
        raise FileNotFoundError(f"Cleaned data not found at {cleaned_data_path}. Please run eda.py first.")
        
    print("--- Loading cleaned dataset ---")
    df = pd.read_csv(cleaned_data_path)
    N = len(df)
    print(f"Loaded {N} records.")
    
    print("\n--- Simulating 12-Month Time Series ---")
    # Initialize arrays of shape (N, 12)
    R_seq = np.zeros((N, 12))
    M_seq = np.zeros((N, 12))
    S_seq = np.zeros((N, 12))
    
    # Month 12 (index 11) is the baseline (actual values)
    R_seq[:, 11] = df["Resource Allocation"].values
    M_seq[:, 11] = df["Mental Fatigue Score"].values
    
    # Satisfaction at month 12
    # Baseline satisfaction is inversely proportional to mental fatigue score plus minor noise
    S_seq[:, 11] = np.clip(10.0 - M_seq[:, 11] + np.random.normal(0, 0.3, N), 0.0, 10.0)
    
    # Simulate backward
    for t in range(10, -1, -1):
        # Resource allocation backward drift (average increase of 0.03/month, with noise)
        drift_R = 0.03
        noise_R = np.random.normal(0, 0.1, N)
        R_seq[:, t] = np.clip(R_seq[:, t+1] - drift_R - noise_R, 1.0, 10.0)
        
        # Mental fatigue backward drift (depends on Resource Allocation + noise)
        drift_M = 0.01 * R_seq[:, t] + 0.01
        noise_M = np.random.normal(0, 0.12, N)
        M_seq[:, t] = np.clip(M_seq[:, t+1] - drift_M - noise_M, 0.0, 10.0)
        
        # Satisfaction is computed based on fatigue with noise
        S_seq[:, t] = np.clip(10.0 - M_seq[:, t] + np.random.normal(0, 0.3, N), 0.0, 10.0)
        
    # Scale satisfaction to 0-1 range to represent a standard ratio
    S_seq = S_seq / 10.0
    
    print("\n--- Engineering New Temporal Features ---")
    volatility_index = np.zeros((N, 12))
    satisfaction_drop = np.zeros((N, 12))
    overtime_trend = np.zeros((N, 12))
    
    S_max = np.zeros(N)
    for t in range(12):
        # 1. Volatility Index: std of Mental Fatigue Score up to month t
        if t == 0:
            volatility_index[:, 0] = 0.0
        else:
            volatility_index[:, t] = np.std(M_seq[:, :t+1], axis=1)
            
        # 2. Satisfaction Drop: max drop in satisfaction up to month t
        if t == 0:
            S_max = S_seq[:, 0].copy()
            satisfaction_drop[:, 0] = 0.0
        else:
            S_max = np.maximum(S_max, S_seq[:, t])
            satisfaction_drop[:, t] = np.maximum(satisfaction_drop[:, t-1], S_max - S_seq[:, t])
            
        # 3. Overtime Trend: slope of Resource Allocation over the last 6 months
        start_idx = max(0, t - 5)
        window_len = t - start_idx + 1
        if window_len <= 1:
            overtime_trend[:, t] = 0.0
        else:
            x = np.arange(window_len)
            x_mean = np.mean(x)
            x_var = np.var(x) * window_len
            
            y = R_seq[:, start_idx:t+1]
            y_mean = np.mean(y, axis=1, keepdims=True)
            cov = np.sum((x - x_mean) * (y - y_mean), axis=1)
            overtime_trend[:, t] = cov / x_var
            
    print("\n--- Constructing Final Feature Sequences ---")
    # Number of features F = 10
    # 0: Gender_Male, 1: Company_Type_Service, 2: WFH_Setup_Available_Yes
    # 3: Designation, 4: Resource_Allocation, 5: Mental_Fatigue_Score, 6: Satisfaction
    # 7: volatility_index, 8: satisfaction_drop, 9: overtime_trend
    F = 10
    X = np.zeros((N, 12, F))
    
    # Map categoricals to binary features and broadcast
    X[:, :, 0] = (df["Gender"].values == "Male").astype(float)[:, np.newaxis]
    X[:, :, 1] = (df["Company Type"].values == "Service").astype(float)[:, np.newaxis]
    X[:, :, 2] = (df["WFH Setup Available"].values == "Yes").astype(float)[:, np.newaxis]
    X[:, :, 3] = df["Designation"].values[:, np.newaxis]
    
    # Assign dynamic features
    X[:, :, 4] = R_seq
    X[:, :, 5] = M_seq
    X[:, :, 6] = S_seq
    X[:, :, 7] = volatility_index
    X[:, :, 8] = satisfaction_drop
    X[:, :, 9] = overtime_trend
    
    # Target variable y: Burn Rate at Month 12
    y = df["Burn Rate"].values
    
    print("\n--- Splitting Data (Train: 70%, Val: 15%, Test: 15%) ---")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
    print(f"Val shapes   - X: {X_val.shape}, y: {y_val.shape}")
    print(f"Test shapes  - X: {X_test.shape}, y: {y_test.shape}")
    
    print("\n--- Scaling Numeric Features ---")
    # Identify indices of numeric/continuous columns to scale
    # Columns 3 to 9 (inclusive) are normalized/continuous features
    scale_cols = [3, 4, 5, 6, 7, 8, 9]
    
    # Fit scaler on Train split
    # First reshape to 2D
    N_train = X_train.shape[0]
    X_train_2d = X_train.reshape(N_train * 12, F)
    
    scaler = StandardScaler()
    scaler.fit(X_train_2d[:, scale_cols])
    
    # Transform splits
    def scale_dataset(dataset):
        n_samples = dataset.shape[0]
        dataset_2d = dataset.reshape(n_samples * 12, F)
        dataset_2d[:, scale_cols] = scaler.transform(dataset_2d[:, scale_cols])
        return dataset_2d.reshape(n_samples, 12, F)
        
    X_train_scaled = scale_dataset(X_train)
    X_val_scaled = scale_dataset(X_val)
    X_test_scaled = scale_dataset(X_test)
    
    # Save scaler
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved fitted StandardScaler to {scaler_path}")
    
    # Save datasets as numpy arrays
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val_scaled)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    print("Saved all preprocessed numpy arrays to processed_data/")
    
    print("--- Preprocessing step completed successfully ---")

if __name__ == "__main__":
    generate_features()
