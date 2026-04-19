"""
train_lstm.py  –  Trening LSTM na datasecie Google ASL Signs
═════════════════════════════════════════════════════════════

Zakłada że load_asl_google.py już skonwertował dane do:
    dataset_pjm/
        book/
            seq_0000.npy   # (30, 63) float32
        drink/
            ...

Użycie:
    python train_lstm.py                          # domyślne ustawienia
    python train_lstm.py --data dataset_pjm       # inny folder
    python train_lstm.py --epochs 100 --batch 32  # inne hiperparametry
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── KONFIGURACJA ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30    # klatki na sekwencję — MUSI być identyczne jak w loaderze
NUM_FEATURES    = 63    # 21 landmarków × 3 — MUSI być identyczne jak w loaderze
MODEL_FILE      = "pjm_lstm_model.keras"
LABEL_MAP_FILE  = "label_map_pjm.json"


# ── KROK 1: WCZYTYWANIE DANYCH ────────────────────────────────────────────────

def load_npy_dataset(data_path: str):
    """
    Skanuje folder dataset_pjm/ i wczytuje wszystkie pliki .npy.

    Zwraca:
      X: numpy array kształtu (N, 30, 63) — N próbek
      y: numpy array etykiet jako stringi (N,)
    """
    X_list, y_list = [], []

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Brak folderu '{data_path}'.\n"
            "Uruchom najpierw: python load_asl_google.py"
        )

    sign_dirs = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and not d.startswith("_")
    ])

    if not sign_dirs:
        raise RuntimeError(f"Brak podfolderów w '{data_path}'.")

    print(f"Znalezione klasy ({len(sign_dirs)}): {sign_dirs}\n")

    for sign in sign_dirs:
        sign_dir = os.path.join(data_path, sign)
        npy_files = [f for f in sorted(os.listdir(sign_dir)) if f.endswith(".npy")]

        loaded = 0
        for fname in npy_files:
            seq = np.load(os.path.join(sign_dir, fname))

            # Defensywne sprawdzenie kształtu
            if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                print(f"  WARN: {sign}/{fname} ma kształt {seq.shape}, pomijam.")
                continue

            X_list.append(seq)
            y_list.append(sign)
            loaded += 1

        print(f"  [{sign}]: {loaded} sekwencji")

    if not X_list:
        raise RuntimeError("Nie wczytano żadnych danych!")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=object)

    print(f"\nDataset: {X.shape[0]} próbek, kształt: {X.shape}")
    return X, y


# ── KROK 2: ENKODOWANIE ETYKIET ───────────────────────────────────────────────

def encode_labels(y_raw: np.ndarray):
    """
    Konwertuje etykiety tekstowe → one-hot encoding.

    Przykład:
      ['book', 'drink', 'book'] → [0, 1, 0] → [[1,0], [0,1], [1,0]]

    Zapisuje mapę indeks→etykieta do JSON (potrzebne w predict.py).
    """
    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)   # 'book'→0, 'drink'→1, ...
    y_cat = to_categorical(y_int)     # 0→[1,0,0,...], 1→[0,1,0,...] (one-hot)

    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"\nMapa etykiet zapisana → {LABEL_MAP_FILE}")
    print(f"  Klasy: {label_map}\n")

    return le, y_int, y_cat


# ── KROK 3: BUDOWA MODELU LSTM ────────────────────────────────────────────────

def build_lstm_model(num_classes: int) -> tf.keras.Model:
    """
    Buduje sieć LSTM zgodną z dokumentacją techniczną projektu.

    Architektura (layer po layerze):

    Input (30, 63)
    │   Tensor wejściowy: 30 klatek, 63 cechy na klatkę.
    │   Keras MUSI znać kształt wejścia żeby zbudować sieć.
    │
    LSTM(64, return_sequences=True, activation='relu')
    │   64 komórki LSTM. Każda komórka ma:
    │     - forget gate:  "co zapomnieć z poprzedniego stanu"
    │     - input gate:   "co zapamiętać z bieżącej klatki"
    │     - output gate:  "co przekazać dalej"
    │   return_sequences=True → zwraca wyjście dla KAŻDEJ klatki
    │   Kształt wyjścia: (30, 64)
    │
    Dropout(0.2)
    │   20% neuronów losowo zerowanych podczas treningu.
    │   Zapobiega overfittingowi (zbyt dokładnemu dopasowaniu do danych treningowych).
    │
    LSTM(128, return_sequences=False, activation='relu')
    │   128 komórek LSTM.
    │   return_sequences=False → zwraca TYLKO wyjście ostatniej klatki.
    │   Agreguje całą sekwencję do wektora 128-elementowego.
    │   Kształt wyjścia: (128,)
    │
    Dense(64, activation='relu')
    Dense(32, activation='relu')
    │   Klasyczne warstwy gęste — nieliniowa klasyfikacja cech.
    │
    Dense(num_classes, activation='softmax')
        Wyjście: wektor prawdopodobieństw dla każdej klasy.
        Softmax gwarantuje że suma = 1.0.
        np. dla 20 klas: [0.01, 0.85, 0.02, ...] → klasa 1 z 85% pewnością
    """
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

        LSTM(64, return_sequences=True, activation="relu"),
        Dropout(0.2),

        LSTM(128, return_sequences=False, activation="relu"),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",                   # adaptacyjny krok uczenia
        loss="categorical_crossentropy",    # strata dla klasyfikacji multi-class
        metrics=["categorical_accuracy"],
    )

    return model


# ── KROK 4: TRENING ───────────────────────────────────────────────────────────

def train(data_path: str, epochs: int, batch_size: int):
    # 1. Dane
    X, y_raw = load_npy_dataset(data_path)
    le, y_int, y_cat = encode_labels(y_raw)
    num_classes = len(le.classes_)

    # 2. Podział train/test
    X_train, X_test, y_train, y_test, y_int_train, y_int_test = train_test_split(
        X, y_cat, y_int,
        test_size=0.2,
        random_state=42,
        stratify=y_int   # równa reprezentacja każdej klasy w obu zbiorach
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Model
    print(f"\nBuduję model LSTM ({num_classes} klas)...")
    model = build_lstm_model(num_classes)
    model.summary()

    # 4. Callbacks
    callbacks = [
        # Przerwij trening jeśli val_loss nie spada przez 10 epok
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Zapisuj najlepszy model w trakcie treningu
        ModelCheckpoint(
            MODEL_FILE,
            monitor="val_categorical_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    # 5. Trening
    print(f"\nRozpoczęcie treningu ({epochs} epok max, batch={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # 6. Ewaluacja
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'═'*40}")
    print(f"Test accuracy: {acc*100:.2f}%")
    print(f"Test loss:     {loss:.4f}")
    print(f"Model zapisany → {MODEL_FILE}")

    return history


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening LSTM na datasecie ASL Signs")
    parser.add_argument("--data",   default="dataset_pjm",
                        help="Folder z .npy sekwencjami")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maksymalna liczba epok (domyślnie 100, EarlyStopping może skrócić)")
    parser.add_argument("--batch",  type=int, default=32,
                        help="Rozmiar batcha (domyślnie 32)")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch)
