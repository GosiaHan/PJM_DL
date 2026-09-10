import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── STAŁE ────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30    # ile klatek chcemy na wyjściu
NUM_LANDMARKS   = 21    # punkty na jedną dłoń (MediaPipe Hands)
NUM_FEATURES    = NUM_LANDMARKS * 3 
HAND_PRIORITY   = ["right_hand", "left_hand"]   # prawa ręka priorytet


# ── KROK 1: parsowanie jednego pliku .parquet ─────────────────────────────────

def parquet_to_hand_sequence(parquet_path: str) -> np.ndarray | None:
    """
    Wczytuje jeden plik .parquet i ekstrahuje sekwencję cech dłoni.

    Zwraca numpy array kształtu (30, 63) lub None jeśli plik nieprawidłowy.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return None

    # Wymagane kolumny
    required = {"frame", "type", "landmark_index", "x", "y", "z"}
    if not required.issubset(df.columns):
        print(f"Missing columns in {parquet_path}: {df.columns}")
        return None

    frames = sorted(df["frame"].unique())
    if len(frames) == 0:
        print(f"No frames in {parquet_path}")
        return None

    raw_sequence = []   # lista wektorów (63,), jeden na klatkę

    for frame_num in frames:
        frame_df = df[df["frame"] == frame_num]
        features = _extract_hand_features(frame_df)
        raw_sequence.append(features)

    if not raw_sequence:
        print(f"Empty raw_sequence for {parquet_path}")
        return None

    raw = np.array(raw_sequence, dtype=np.float32)  # (T, 63)
    return _resample_sequence(raw, SEQUENCE_LENGTH)   # (30, 63)


def _extract_hand_features(frame_df: pd.DataFrame) -> np.ndarray:
    """
    Z wierszy jednej klatki wyciąga 63-elementowy wektor cech dłoni.

    Logika wyboru ręki:
      - Sprawdza prawą rękę (right_hand), potem lewą (left_hand)
      - Jeśli wszystkie wartości x/y/z są NaN → przechodzi do następnej
      - Jeśli żadna ręka nie ma danych → zwraca wektor zer (zero-padding)

    Format wyjściowy: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    Kolejność odpowiada numeracji MediaPipe Hands (landmark_index 0-20).
    """
    for hand_type in HAND_PRIORITY:
        hand_df = (frame_df[frame_df["type"] == hand_type]
                   .sort_values("landmark_index")
                   .head(NUM_LANDMARKS))

        if len(hand_df) == 0:
            continue

        xs = hand_df["x"].values.astype(np.float32)
        ys = hand_df["y"].values.astype(np.float32)
        zs = hand_df["z"].values.astype(np.float32)

        # Brak detekcji ręki = wszystkie NaN
        if np.isnan(xs).all():
            continue


        xs = np.nan_to_num(xs, nan=0.0)
        ys = np.nan_to_num(ys, nan=0.0)
        zs = np.nan_to_num(zs, nan=0.0)

        # Pad do 21 jeśli mniej punktów (rzadkie, ale defensywne)
        if len(xs) < NUM_LANDMARKS:
            pad = NUM_LANDMARKS - len(xs)
            xs = np.concatenate([xs, np.zeros(pad)])
            ys = np.concatenate([ys, np.zeros(pad)])
            zs = np.concatenate([zs, np.zeros(pad)])

       
        features = np.empty(NUM_FEATURES, dtype=np.float32)
        for i in range(NUM_LANDMARKS):
            features[i * 3]     = xs[i]
            features[i * 3 + 1] = ys[i]
            features[i * 3 + 2] = zs[i]

        return features

    # Żadna ręka zero-padding
    return np.zeros(NUM_FEATURES, dtype=np.float32)


def _resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resamplinguje sekwencję dowolnej długości do target_len klatek.

    Metoda: interpolacja liniowa wzdłuż osi czasu.
    - Jeśli seq ma 15 klatek a chcemy 30 → każda klatka pojawia się
      "pomiędzy" oryginalnymi, interpolujemy liniowo.
    - Jeśli seq ma 60 klatek a chcemy 30 → co druga klatka.

    Jest to prostsze niż FFT resample, ale wystarczające dla danych
    o stosunkowo równomiernym tempie (nagrania znaków migowych).
    """
    T = len(seq)
    if T == target_len:
        return seq

    
    indices = np.linspace(0, T - 1, target_len)

    resampled = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for new_i, idx in enumerate(indices):
        lo = int(idx)
        hi = min(lo + 1, T - 1)
        alpha = idx - lo   
        resampled[new_i] = seq[lo] * (1 - alpha) + seq[hi] * alpha

    return resampled




def convert_dataset(src_dir: str, out_dir: str, max_classes: int | None):
    """
    Główna funkcja konwertująca Google ASL Signs → dataset_pjm/

    Wczytuje train.csv, dla każdego znaku (sign) przetwarza wszystkie
    pliki .parquet i zapisuje sekwencje jako:
        out_dir/<sign>/seq_000.npy
        out_dir/<sign>/seq_001.npy
        ...
    """
    train_csv   = os.path.join(src_dir, "train.csv")
    landmark_dir = src_dir 

    # Walidacja
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"\nNie znaleziono: {train_csv}\n\n"
            "Pobierz dataset:\n"
            "  pip install kaggle\n"
            "  kaggle competitions download -c asl-signs\n"
            "  unzip asl-signs.zip -d asl_signs/\n"
        )

    df = pd.read_csv(train_csv)
    print(f"train.csv: {len(df)} próbek, kolumny: {list(df.columns)}\n")

    # Wybór klas
    all_signs = sorted(df["sign"].unique())
    if max_classes:
        selected = all_signs[:max_classes]
        df = df[df["sign"].isin(selected)]
        print(f"Wybrano {len(selected)} z {len(all_signs)} klas: {selected}\n")
    else:
        print(f"Przetwarzam wszystkie {len(all_signs)} klas.\n")

    os.makedirs(out_dir, exist_ok=True)

    total_ok = 0
    total_fail = 0

    for sign in sorted(df["sign"].unique()):
        sign_df  = df[df["sign"] == sign]
        sign_out = os.path.join(out_dir, sign)
        os.makedirs(sign_out, exist_ok=True)

        ok = 0
        for i, (_, row) in enumerate(tqdm(sign_df.iterrows(),
                                          total=len(sign_df),
                                          desc=f"{sign:15s}",
                                          leave=True)):

            parquet_path = os.path.join(landmark_dir, row["path"].replace("train_landmark_files/", ""))

            if not os.path.exists(parquet_path):
                total_fail += 1
                continue

            seq = parquet_to_hand_sequence(parquet_path)

            if seq is None:
                total_fail += 1
                continue
            if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                print(f"Wrong shape: {seq.shape}, expected {(SEQUENCE_LENGTH, NUM_FEATURES)}")
                total_fail += 1
                continue

            out_path = os.path.join(sign_out, f"seq_{i:04d}.npy")
            np.save(out_path, seq)
            ok += 1
            total_ok += 1

        print(f"  → [{sign}] zapisano {ok}/{len(sign_df)} sekwencji")

    # Zapisz metadane (opcjonalne, ale pomocne)
    meta = {
        "sequence_length": SEQUENCE_LENGTH,
        "num_features": NUM_FEATURES,
        "total_sequences": total_ok,
        "failed": total_fail,
        "signs": sorted(df["sign"].unique().tolist()),
    }
    meta_path = os.path.join(out_dir, "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'═'*50}")
    print(f"Gotowe!  OK: {total_ok}  |  Pominięto: {total_fail}")
    print(f"Dataset: {out_dir}/")
    print(f"Meta:    {meta_path}")
    print(f"\nNastępny krok:  python train_lstm.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Google ASL Signs (Kaggle) → .npy sekwencje landmarków",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--src", default="asl_signs",
        help="Folder z rozpakowanym datasetem (zawiera train.csv)"
    )
    parser.add_argument(
        "--out", default="dataset_pjm",
        help="Folder wyjściowy na sekwencje .npy (domyślnie: dataset_pjm)"
    )
    parser.add_argument(
        "--classes", type=int, default=20,
        help="Ile znaków przetworzyć (domyślnie: 20, max: 250). "
             "Zacznij od 5-10 żeby sprawdzić czy działa."
    )
    args = parser.parse_args()

    convert_dataset(args.src, args.out, args.classes)
