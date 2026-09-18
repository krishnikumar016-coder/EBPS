import sys
import os

# Add the src directory to Python's system path for easy imports
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(src_path)

def main():
    print("==================================================")
    print("STARTING EMPLOYEE BURNOUT PREPROCESSING PIPELINE")
    print("==================================================")
    
    # 1. Download Data
    print("\n--- STEP 1: DOWNLOADING DATASETS ---")
    import download
    download.main()
    
    # 2. Perform EDA
    print("\n--- STEP 2: RUNNING EXPLORATORY DATA ANALYSIS ---")
    import eda
    eda.perform_eda()
    
    # 3. Perform Feature Engineering & Preprocessing
    print("\n--- STEP 3: RUNNING FEATURE ENGINEERING & DATA SPLITTING ---")
    import features
    features.generate_features()
    
    print("\n==================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
