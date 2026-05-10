"""
train_lstm.py – Training LSTM on Google ASL Signs dataset converted to .npy

Pipeline:
1. Load .npy sequences from dataset_pjm/
2. Encode labels
3. Split off independent test set
4. Perform Stratified K-Fold validation on training data
5. Train final model using average best epoch from K-Fold
6. Evaluate once on test set
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63
MODEL_FILE = "pjm_lstm_model.keras"
LABEL_MAP_FILE = "label_map_pjm.json"


# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────

def load_npy_dataset(data_path: str):
    X_list, y_list = [], []

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Folder '{data_path}' not found. Run load_asl_google.py first."
        )

    sign_dirs = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and not d.startswith("_")
    ])

    if not sign_dirs:
        raise RuntimeError(f"No class folders found in '{data_path}'.")

    print(f"Found classes ({len(sign_dirs)}): {sign_dirs}\n")

    for sign in sign_dirs:
        sign_dir = os.path.join(data_path, sign)
        npy_files = [f for f in sorted(os.listdir(sign_dir)) if f.endswith(".npy")]

        loaded = 0

        for fname in npy_files:
            seq = np.load(os.path.join(sign_dir, fname))

            if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                print(f"WARNING: {sign}/{fname} shape {seq.shape}, skipped.")
                continue

            X_list.append(seq)
            y_list.append(sign)
            loaded += 1

        print(f"[{sign}]: {loaded} sequences")

    if not X_list:
        raise RuntimeError("No data loaded.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=object)

    print(f"\nDataset: {X.shape[0]} samples, shape: {X.shape}")
    return X, y


# ── STEP 2: ENCODE LABELS ─────────────────────────────────────────────────────

def encode_labels(y_raw: np.ndarray):
    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    y_cat = to_categorical(y_int)

    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}

    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"\nLabel map saved → {LABEL_MAP_FILE}")
    print(f"Classes: {label_map}\n")

    return le, y_int, y_cat


# ── STEP 3: BUILD LSTM MODEL ──────────────────────────────────────────────────
# CHANGE REQUESTED BY INSTRUCTOR:
# Earlier version used activation="relu" inside LSTM layers.
# This was changed to default LSTM activation (tanh), which is more suitable
# for sequence modelling. Accuracy improved from about 67.73% to 72.91%
# on the 20-class experiment.

def build_lstm_model(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

        LSTM(64, return_sequences=True),
        Dropout(0.3),

        LSTM(128, return_sequences=False),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model


# ── STEP 4: TRAINING WITH K-FOLD VALIDATION ───────────────────────────────────

def train(data_path: str, epochs: int, batch_size: int):
    # 1. Load data
    X, y_raw = load_npy_dataset(data_path)
    le, y_int, y_cat = encode_labels(y_raw)
    num_classes = len(le.classes_)

    # 2. Separate final test set
    # The test set is NOT used during K-Fold validation.
    X_train_all, X_test, y_train_all, y_test, y_int_train_all, y_int_test = train_test_split(
        X,
        y_cat,
        y_int,
        test_size=0.15,
        random_state=42,
        stratify=y_int
    )

    print(f"Train + validation: {len(X_train_all)} | Test: {len(X_test)}")

    # 3. Stratified K-Fold validation
    # This is the only validation stage.
    n_splits = 5
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    fold_accuracies = []
    fold_losses = []
    fold_best_epochs = []

    print(f"\nStarting Stratified K-Fold validation: {n_splits} folds\n")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train_all, y_int_train_all),
        start=1
    ):
        print(f"\n{'═' * 50}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'═' * 50}")

        X_train = X_train_all[train_idx]
        X_val = X_train_all[val_idx]

        y_train = y_train_all[train_idx]
        y_val = y_train_all[val_idx]

        print(f"Train: {len(X_train)} | Validation: {len(X_val)}")

        model = build_lstm_model(num_classes)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

        fold_losses.append(val_loss)
        fold_accuracies.append(val_acc)
        fold_best_epochs.append(best_epoch)

        print(f"\nFold {fold} validation accuracy: {val_acc * 100:.2f}%")
        print(f"Fold {fold} validation loss:     {val_loss:.4f}")
        print(f"Best epoch in fold {fold}:       {best_epoch}")

    # 4. K-Fold summary
    avg_best_epoch = int(round(np.mean(fold_best_epochs)))

    print(f"\n{'═' * 50}")
    print("K-FOLD VALIDATION RESULTS")
    print(f"{'═' * 50}")
    print(f"Average validation accuracy: {np.mean(fold_accuracies) * 100:.2f}%")
    print(f"Standard deviation:          {np.std(fold_accuracies) * 100:.2f}%")
    print(f"Average validation loss:     {np.mean(fold_losses):.4f}")
    print(f"Average best epoch:          {avg_best_epoch}")

    # 5. Final model training
    # No validation is used here, so validation does NOT happen twice.
    # The number of epochs is chosen based on K-Fold results.
    print(f"\n{'═' * 50}")
    print("FINAL MODEL TRAINING")
    print(f"{'═' * 50}")

    final_model = build_lstm_model(num_classes)
    final_model.summary()

    history = final_model.fit(
        X_train_all,
        y_train_all,
        epochs=avg_best_epoch,
        batch_size=batch_size,
        verbose=1
    )

    final_model.save(MODEL_FILE)

    # 6. Final test evaluation
    # Test set is used only once, at the end.
    test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{'═' * 50}")
    print("FINAL TEST RESULT")
    print(f"{'═' * 50}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Model saved → {MODEL_FILE}")

    return history


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM training on ASL Signs dataset")
    parser.add_argument("--data", default="dataset_pjm", help="Folder with .npy sequences")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch)