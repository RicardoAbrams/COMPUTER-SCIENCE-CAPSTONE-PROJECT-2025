import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

# =========================
# CONFIG
# =========================
DATASET_ROOT = r"C:\Users\vipap\Documents\Data\ASL_Alphabet_Dataset\asl_alphabet_train"
OUTPUT_ROOT = Path("templates_data")
OUTPUT_ROOT.mkdir(exist_ok=True)

MAX_PER_LABEL = None  # None = todas
MIN_CONF = 0.5

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

def vectorize_landmarks(hand_landmarks):
    """
    Convierte 21 landmarks -> vector 63
    Normalización idéntica a tu backend
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float32)

    # centrar en muñeca (0)
    pts = pts - pts[0]

    v = pts.reshape(-1)  # 63
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    labels = sorted([d for d in os.listdir(DATASET_ROOT)
                     if os.path.isdir(os.path.join(DATASET_ROOT, d))])

    total_saved = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=MIN_CONF
    ) as hands:

        for label in labels:
            label_path = os.path.join(DATASET_ROOT, label)
            out_dir = OUTPUT_ROOT / label.strip().upper()
            ensure_dir(out_dir)

            files = [f for f in os.listdir(label_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            if MAX_PER_LABEL is not None:
                files = files[:MAX_PER_LABEL]

            saved = 0
            skipped = 0
            print(f"\nProcesando {label} ({len(files)} imágenes)...")

            for fname in files:
                img = cv2.imread(os.path.join(label_path, fname))
                if img is None:
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                if not res.multi_hand_landmarks:
                    skipped += 1
                    continue

                v = vectorize_landmarks(res.multi_hand_landmarks[0])
                np.save(out_dir / f"{saved}.npy", v)

                saved += 1
                total_saved += 1

            print(f"  ✅ guardadas: {saved} | ⚠️ saltadas: {skipped}")

    print(f"\n✅ DONE. Total guardadas: {total_saved}")
    print(f"Output: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()
