import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def build_cnn_lstm_model(input_shape=(12, 10)):
    """
    Builds the hybrid CNN-LSTM architecture using TensorFlow Keras.
    Combine Conv1D layers (for spatial/local pattern extraction) with an LSTM layer
    (for modeling temporal dependencies over the 12-month period).
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        LSTM(units=64, return_sequences=False),
        Dropout(rate=0.2),
        Dense(units=1, activation='sigmoid')
    ])
    return model

def evaluate_predictions(y_true, y_pred_prob, threshold=0.5):
    """
    Computes key performance metrics (Accuracy, F1-Score, ROC-AUC) on predicted probabilities.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    
    print("\n==========================================")
    print("TEST SET EVALUATION METRICS")
    print("==========================================")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {auc:.4f} ({auc*100:.2f}%)")
    print("==========================================")
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return {"accuracy": acc, "f1_score": f1, "roc_auc": auc, "confusion_matrix": cm}

def plot_training_history(history, plots_dir):
    """
    Generates subplots of the training/validation loss and accuracy curves.
    """
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    axes[0].plot(history.history['loss'], label='Train Loss', color='#1f77b4', lw=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True)
    
    # Accuracy Curve
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', lw=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', lw=2)
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training history curves to {plot_path}")

def plot_evaluation_curves(y_true, y_pred_prob, plots_dir, threshold=0.5):
    """
    Generates subplots of the ROC curve and the confusion matrix heatmap.
    """
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    axes[0].plot(fpr, tpr, color='#d62728', lw=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(True)
    
    # 2. Confusion Matrix Heatmap
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xticklabels(['Low Risk (0)', 'High Risk (1)'], fontsize=11)
    axes[1].set_yticklabels(['Low Risk (0)', 'High Risk (1)'], fontsize=11)
    axes[1].set_title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "evaluation_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved evaluation plots to {plot_path}")
