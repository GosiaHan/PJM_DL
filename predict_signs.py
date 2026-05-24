"""
predict_signs.py

Usage:
    python predict_signs.py --source "C:\path\to\video.mp4"
    python predict_signs.py --source "C:\path\to\video.mp4" --threshold 0.6
    python predict_signs.py --source "C:\path\to\video.mp4" --save output.mp4

On first run the hand detection model (~25 MB) is downloaded automatically
and saved as hand_landmarker.task next to this script. After that it works offline.
"""

import os, sys, json, collections, time, argparse, urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf

# ── Settings ───────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH   = 30
NUM_FEATURES      = 63
MODEL_FILE        = "pjm_lstm_model.keras"
LABEL_MAP_FILE    = "label_map_pjm.json"
HAND_MODEL_FILE   = "hand_landmarker.task"
HAND_MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
    (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]

# ── Auto-download hand model ───────────────────────────────────────────────────
def ensure_hand_model():
    if os.path.exists(HAND_MODEL_FILE):
        return
    print("Hand detection model not found — downloading (~25 MB, one time only)...")
    try:
        def progress(block, block_size, total):
            done = block * block_size
            if total > 0:
                pct = min(done / total * 100, 100)
                print(f"\r  {pct:.0f}%", end="", flush=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_FILE, reporthook=progress)
        print(f"\nSaved as '{HAND_MODEL_FILE}'.")
    except Exception as e:
        sys.exit(f"\n[ERROR] Download failed: {e}")

# ── Load resources ─────────────────────────────────────────────────────────────
def load_label_map():
    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}

def load_keras_model():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_FILE)
    print("Model ready.")
    return model

def create_landmarker():
    base = mp_python.BaseOptions(model_asset_path=HAND_MODEL_FILE)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

# ── Landmark processing ────────────────────────────────────────────────────────
def get_features(landmarks):
    """landmarks: list of 21 NormalizedLandmark from Tasks API"""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0]
    scale = np.max(np.abs(coords))
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()

def draw_hand(frame, landmarks, H, W):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 64), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)

# ── HUD ────────────────────────────────────────────────────────────────────────
def draw_hud(frame, label, conf, threshold, buf_len, fps):
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (W, 65), (20, 20, 20), -1)
    color = (0, 220, 80) if conf and conf >= threshold else (0, 200, 255)
    cv2.putText(frame, label.upper() if label else "---",
                (14, 46), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2)
    if conf is not None:
        bw = int(W * 0.35)
        cv2.rectangle(frame, (14, 54), (14+bw, 62), (60,60,60), -1)
        cv2.rectangle(frame, (14, 54), (14+int(bw*conf), 62), color, -1)
        cv2.putText(frame, f"{conf*100:.1f}%", (14+bw+6, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
    cv2.rectangle(frame, (0, H-28), (W, H), (20,20,20), -1)
    cv2.rectangle(frame, (14, H-18), (194, H-8), (60,60,60), -1)
    cv2.rectangle(frame, (14, H-18), (14+int(180*buf_len/SEQUENCE_LENGTH), H-8), (0,220,80), -1)
    cv2.putText(frame, f"buf {buf_len}/{SEQUENCE_LENGTH}  fps {fps:.0f}",
                (200, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save",      default=None)
    args = parser.parse_args()

    ensure_hand_model()
    label_map  = load_label_map()
    model      = load_keras_model()
    landmarker = create_landmarker()
    print(f"Classes ({len(label_map)}): {list(label_map.values())}\n")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {W}x{H} @ {FPS:.0f} fps  — press Q to quit.\n")

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    sequence  = collections.deque(maxlen=SEQUENCE_LENGTH)
    pred_hist = collections.deque(maxlen=5)
    label     = None
    conf      = None
    no_hand   = 0
    t0        = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind and loop
            continue

        fps_now = 1.0 / max(time.time() - t0, 1e-6)
        t0 = time.time()

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_img)

        if result.hand_landmarks:
            no_hand = 0
            lms = result.hand_landmarks[0]
            draw_hand(frame, lms, H, W)
            sequence.append(get_features(lms))
        else:
            no_hand += 1
            if no_hand > FPS:
                sequence.clear(); pred_hist.clear()
                label = None; conf = None
            cv2.putText(frame, "No hand", (20, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)

        if len(sequence) == SEQUENCE_LENGTH:
            X     = np.expand_dims(np.array(sequence, dtype=np.float32), 0)
            probs = model.predict(X, verbose=0)[0]
            idx   = int(np.argmax(probs))
            pred_hist.append(idx)
            best  = collections.Counter(pred_hist).most_common(1)[0][0]
            label = label_map.get(best, str(best))
            conf  = float(probs[best])

        draw_hud(frame, label, conf, args.threshold, len(sequence), fps_now)
        cv2.imshow("Sign Prediction  [Q = quit]", frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()