"""
train_lstm.py – Training LSTM on Google ASL Signs dataset converted to .npy

Pipeline:
1. Load .npy sequences from dataset_pjm/ (with caching and class selection)
2. Encode labels
3. Split off independent test set
4. Perform Stratified K-Fold validation on training data
5. Train final model using average best epoch from K-Fold
6. Evaluate once on test set
"""

import os
import json
import argparse
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63
MODEL_FILE = "pjm_lstm_model.keras"
LABEL_MAP_FILE = "label_map_pjm.json"
CACHE_DIR = "cache"
DEFAULT_DATA_PATH = "dataset_pjm"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_CLASSES = "all"
DEFAULT_NUM_CLASSES = 0
DEFAULT_USE_CACHE = True
DEFAULT_REFRESH_CACHE = False


def resolve_selected_classes(data_path: str, classes_arg: str, num_classes: int):
    if classes_arg and classes_arg != "all":
        classes = [c.strip() for c in classes_arg.split(",") if c.strip()]
        if not classes:
            raise ValueError("--classes must contain at least one valid class name.")
        return classes

    if num_classes <= 0:
        return None

    sign_dirs = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
        and not d.startswith("_")
        and d != CACHE_DIR
    ])

    if not sign_dirs:
        raise RuntimeError(f"No class folders found in '{data_path}'.")

    if num_classes > len(sign_dirs):
        raise ValueError(
            f"Requested {num_classes} classes, but only {len(sign_dirs)} are available."
        )

    return sign_dirs[:num_classes]


# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────

def get_cache_paths(data_path: str, classes: list | None):
    cache_path = os.path.join(data_path, CACHE_DIR)
    if classes is None:
        cache_key = "all"
    else:
        class_key = ",".join(sorted(classes))
        cache_key = hashlib.md5(class_key.encode("utf-8")).hexdigest()

    return cache_path, os.path.join(cache_path, f"X_{cache_key}.npy"), os.path.join(cache_path, f"y_{cache_key}.npy")


def load_npy_dataset(data_path: str, classes: list = None, use_cache: bool = True, refresh_cache: bool = False):
    cache_path, X_cache, y_cache = get_cache_paths(data_path, classes)
    can_use_cache = use_cache and not refresh_cache and os.path.exists(X_cache) and os.path.exists(y_cache)

    if can_use_cache:
        print("Loading data from cache...")
        X = np.load(X_cache)
        y = np.load(y_cache, allow_pickle=True)
        print(f"Dataset loaded from cache: {X.shape[0]} samples, shape: {X.shape}")
        return X, y

    X_list, y_list = [], []

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Folder '{data_path}' not found. Run load_asl_google.py first."
        )

    sign_dirs = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
        and not d.startswith("_")
        and d != CACHE_DIR
    ])

    if not sign_dirs:
        raise RuntimeError(f"No class folders found in '{data_path}'.")

    if classes and classes != ['all']:
        sign_dirs = [d for d in sign_dirs if d in classes]
        if not sign_dirs:
            raise RuntimeError(f"None of the specified classes found in '{data_path}'.")

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

    # Save to cache
    os.makedirs(cache_path, exist_ok=True)
    np.save(X_cache, X)
    np.save(y_cache, y)
    print(f"Data saved to cache at {cache_path}")

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


def plot_history(history, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["categorical_accuracy"], label="train accuracy")
    if "val_categorical_accuracy" in history.history:
        plt.plot(history.history["val_categorical_accuracy"], label="val accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    path = os.path.join(output_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Training history saved → {path}")


def plot_confusion_matrix(y_true, y_pred, labels, output_dir: str = "plots", normalize: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
        filename = "confusion_matrix_normalized.png"
    else:
        fmt = "d"
        title = "Confusion Matrix"
        filename = "confusion_matrix.png"

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", values_format=fmt)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"{title} saved → {path}")


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

def train(data_path: str, epochs: int, batch_size: int, classes: list = None, use_cache: bool = DEFAULT_USE_CACHE, refresh_cache: bool = DEFAULT_REFRESH_CACHE):
    # 1. Load data
    X, y_raw = load_npy_dataset(data_path, classes, use_cache=use_cache, refresh_cache=refresh_cache)
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

    y_pred_probs = final_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_names = list(le.classes_)

    print(f"\n{'═' * 50}")
    print("FINAL TEST CLASSIFICATION REPORT")
    print(f"{'═' * 50}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    plot_history(history)
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=False)

    return history


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM training on ASL Signs dataset")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Folder with .npy sequences")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--classes", type=str, default=DEFAULT_CLASSES, help="Comma-separated list of classes to use, or 'all' for all classes")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES, help="Number of classes to use from the dataset (uses first sorted classes)")
    parser.add_argument("--class-count", type=int, default=DEFAULT_NUM_CLASSES, help="Alias for --num-classes; number of classes to use")
    parser.add_argument("--cache", dest="use_cache", action="store_true", default=DEFAULT_USE_CACHE, help="Reuse cached loaded data when available")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Do not use the data cache")
    parser.add_argument("--refresh-cache", action="store_true", default=DEFAULT_REFRESH_CACHE, help="Reload raw .npy files and refresh the cache")
    args = parser.parse_args()

    selected_num_classes = args.class_count if args.class_count > 0 else args.num_classes
    selected_classes = resolve_selected_classes(args.data, args.classes, selected_num_classes)

    train(
        args.data,
        args.epochs,
        args.batch,
        selected_classes,
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
    )