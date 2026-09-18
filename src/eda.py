import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda():
    # Set plot styles
    sns.set_theme(style="whitegrid")
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", "train.csv")
    plots_dir = os.path.join(project_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"train.csv not found at {data_path}. Please run download.py first.")
        
    print("--- Loading dataset ---")
    df = pd.read_csv(data_path)
    
    print(f"Initial dataset shape: {df.shape}")
    
    print("\n--- Column information ---")
    print(df.info())
    
    print("\n--- Missing values count ---")
    null_counts = df.isnull().sum()
    print(null_counts)
    
    print("\n--- Missing values percentage ---")
    print(100 * null_counts / len(df))
    
    print("\n--- Basic summary statistics ---")
    print(df.describe(include='all'))
    
    print("\n--- Cleaning dataset (Dropping missing values) ---")
    df_cleaned = df.dropna()
    print(f"Original row count: {len(df)}")
    print(f"Cleaned row count: {len(df_cleaned)}")
    print(f"Dropped {len(df) - len(df_cleaned)} rows ({100 * (len(df) - len(df_cleaned)) / len(df):.2f}%)")
    
    # Save cleaned data to SWE_PROJECT/data
    cleaned_path = os.path.join(project_dir, "data", "train_cleaned.csv")
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_path}")
    
    # Visualize distributions
    numeric_cols = ["Designation", "Resource Allocation", "Mental Fatigue Score", "Burn Rate"]
    
    print("\n--- Generating histograms ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, col in enumerate(numeric_cols):
        ax = axes[i // 2, i % 2]
        sns.histplot(df_cleaned[col], kde=True, ax=ax, color='#1f77b4', bins=20)
        ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()
    hist_plot_path = os.path.join(plots_dir, "numerical_distributions.png")
    plt.savefig(hist_plot_path, dpi=150)
    plt.close()
    print(f"Saved distribution histograms to {hist_plot_path}")
    
    print("\n--- Generating box plots ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, col in enumerate(numeric_cols):
        ax = axes[i // 2, i % 2]
        sns.boxplot(y=df_cleaned[col], ax=ax, color='#ff7f0e')
        ax.set_title(f'Boxplot of {col}', fontsize=14, fontweight='bold')
        ax.set_ylabel(col, fontsize=12)
    plt.tight_layout()
    box_plot_path = os.path.join(plots_dir, "numerical_boxplots.png")
    plt.savefig(box_plot_path, dpi=150)
    plt.close()
    print(f"Saved boxplots to {box_plot_path}")
    
    print("\n--- Generating correlation heatmap ---")
    plt.figure(figsize=(8, 6))
    corr = df_cleaned[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".3f", annot_kws={"size": 12})
    plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_plot_path = os.path.join(plots_dir, "correlation_heatmap.png")
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to {corr_plot_path}")
    print("--- EDA step completed successfully ---")

if __name__ == "__main__":
    perform_eda()
