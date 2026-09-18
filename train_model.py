import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# Add src folder to system path for local module loading
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import build_cnn_lstm_model, evaluate_predictions, plot_training_history, plot_evaluation_curves

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, "processed_data")
    plots_dir = os.path.join(project_dir, "plots")
    
    # Load processed data
    print("--- Loading datasets ---")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train_raw = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val_raw = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test_raw = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    # Binarize targets (threshold >= 0.5 represents High Risk)
    print("--- Binarizing targets (threshold >= 0.5) ---")
    y_train = (y_train_raw >= 0.5).astype(int)
    y_val = (y_val_raw >= 0.5).astype(int)
    y_test = (y_test_raw >= 0.5).astype(int)
    
    # Print class balance to check distribution
    train_balance = np.mean(y_train) * 100
    val_balance = np.mean(y_val) * 100
    test_balance = np.mean(y_test) * 100
    print(f"High risk class (1) percentage - Train: {train_balance:.2f}%, Val: {val_balance:.2f}%, Test: {test_balance:.2f}%")
    
    # Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"--- Building CNN-LSTM model with input shape {input_shape} ---")
    model = build_cnn_lstm_model(input_shape)
    
    model.summary()
    
    # Compile Model
    print("--- Compiling model ---")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    model_path = os.path.join(project_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # Train
    print("--- Starting training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history curves
    plot_training_history(history, plots_dir)
    
    # Load best checkpoint model to evaluate
    print(f"--- Loading best model from {model_path} ---")
    best_model = load_model(model_path)
    
    # Predict on test set
    print("--- Running predictions on test set ---")
    y_pred_prob = best_model.predict(X_test).flatten()
    
    # Evaluate predictions
    metrics = evaluate_predictions(y_test, y_pred_prob, threshold=0.5)
    
    # Save evaluation plots (ROC and Confusion Matrix Heatmap)
    plot_evaluation_curves(y_test, y_pred_prob, plots_dir, threshold=0.5)
    
    print("\n--- Model training and evaluation finished successfully ---")

def retrain_model(epochs=15, batch_size=64, learning_rate=0.001):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, "processed_data")
    plots_dir = os.path.join(project_dir, "plots")
    
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train_raw = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val_raw = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test_raw = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    y_train = (y_train_raw >= 0.5).astype(int)
    y_val = (y_val_raw >= 0.5).astype(int)
    y_test = (y_test_raw >= 0.5).astype(int)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model_path = os.path.join(project_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    plot_training_history(history, plots_dir)
    best_model = load_model(model_path)
    y_pred_prob = best_model.predict(X_test, verbose=0).flatten()
    metrics = evaluate_predictions(y_test, y_pred_prob, threshold=0.5)
    plot_evaluation_curves(y_test, y_pred_prob, plots_dir, threshold=0.5)
    
    # Reset singleton in predict module if loaded
    try:
        import api.predict as predict_module
        predict_module._model = None
    except Exception:
        pass

    return {
        "status": "success",
        "message": f"CNN-LSTM Model successfully retrained for {epochs} epochs.",
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
        "metrics": metrics
    }

if __name__ == "__main__":
    main()

