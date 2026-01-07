# -*- coding: utf-8 -*-

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import uvicorn
import numpy as np
import cv2
import math
import os
import json
import time
import uuid

# ✅ Import estable en Windows
from mediapipe.python.solutions import hands as mp_hands


# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates_data")   # samples por letra
TEMPLATES_OUT = os.path.join(BASE_DIR, "templates.json")   # templates promedio

# Umbral: mientras MÁS pequeño, más estricto.
# Si te da muchos None, sube a 3.0–4.0
# Si te confunde letras, baja a 2.0–2.5
MATCH_THRESHOLD = 3.0


# =========================
# App init
# =========================
app = FastAPI(title="SignalSpeak Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev; luego limita a tu host/puerto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hands_detector = None
templates_cache = None  # {"A": [..features..], ...}
templates_counts = None # {"A": 120, ...}


# =========================
# Helpers: templates I/O
# =========================
def load_templates_if_exist():
    global templates_cache, templates_counts

    if not os.path.exists(TEMPLATES_OUT):
        templates_cache = None
        templates_counts = None
        return False

    try:
        with open(TEMPLATES_OUT, "r", encoding="utf-8") as f:
            data = json.load(f)

        templates_cache = data.get("templates", None)
        templates_counts = data.get("counts", None)

        # sanity
        if not isinstance(templates_cache, dict) or len(templates_cache) == 0:
            templates_cache = None
            templates_counts = None
            return False

        return True
    except:
        templates_cache = None
        templates_counts = None
        return False


def ensure_letter_dirs():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    for c in range(ord("A"), ord("Z") + 1):
        os.makedirs(os.path.join(TEMPLATES_DIR, chr(c)), exist_ok=True)


# =========================
# Helpers: landmarks -> features
# =========================
def normalize_landmarks_xy(lm):
    """
    Normaliza para que no dependa tanto de zoom/posición:
    - centra en wrist (0)
    - escala por distancia wrist(0) -> middle_mcp(9)
    - usa solo x,y para estabilidad
    """
    wrist = lm[0]
    ref = lm[9]  # middle_mcp
    scale = math.sqrt((ref["x"] - wrist["x"]) ** 2 + (ref["y"] - wrist["y"]) ** 2) + 1e-9

    out = []
    for p in lm:
        out.append({
            "x": (p["x"] - wrist["x"]) / scale,
            "y": (p["y"] - wrist["y"]) / scale,
        })
    return out


def features_from_norm_lm(nlm):
    """
    Features simples y efectivos:
    - concatenación [x1,y1,x2,y2,...] => 42 valores
    """
    feat = []
    for p in nlm:
        feat.append(float(p["x"]))
        feat.append(float(p["y"]))
    return feat


def euclidean(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def predict_from_templates(feat):
    """
    Template matching:
    - compara con cada letra (distancia euclidiana)
    - devuelve la letra con menor distancia si pasa umbral
    """
    if not templates_cache:
        return None, None

    best_letter = None
    best_score = None

    for letter, templ in templates_cache.items():
        d = euclidean(feat, templ)
        if best_score is None or d < best_score:
            best_score = d
            best_letter = letter

    if best_score is None:
        return None, None

    # aplica umbral para evitar falsos positivos
    if best_score <= MATCH_THRESHOLD:
        return best_letter, best_score

    return None, best_score


# =========================
# Startup / shutdown
# =========================
@app.on_event("startup")
def startup():
    global hands_detector
    ensure_letter_dirs()

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    load_templates_if_exist()

    print("✅ MediaPipe Hands initialized")
    if templates_cache:
        print(f"✅ templates.json loaded with {len(templates_cache)} letters")
    else:
        print("ℹ️ templates.json not found (collect samples then build templates)")


@app.on_event("shutdown")
def shutdown():
    global hands_detector
    if hands_detector:
        hands_detector.close()
        print("🛑 MediaPipe Hands closed")


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SignalSpeak Python backend active",
        "templates_loaded": bool(templates_cache),
        "match_threshold": MATCH_THRESHOLD,
    }


@app.post("/collect-template")
async def collect_template(label: str, file: UploadFile = File(...)):
    """
    Guarda una muestra (features) en templates_data/<label>/xxxx.json
    Uso:
      POST /collect-template?label=C  (multipart file=frame.jpg)
    """
    label = (label or "").strip().upper()
    if len(label) != 1 or not ("A" <= label <= "Z"):
        return JSONResponse({"error": "label must be A-Z"}, status_code=400)

    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)

    if not result.multi_hand_landmarks:
        return {"saved": False, "reason": "no_hand_detected", "label": label}

    # primera mano
    hand_landmarks = result.multi_hand_landmarks[0]
    lm_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_landmarks.landmark]

    nlm = normalize_landmarks_xy(lm_list)
    feat = features_from_norm_lm(nlm)

    out_dir = os.path.join(TEMPLATES_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
    out_path = os.path.join(out_dir, fname)

    payload = {
        "label": label,
        "features": feat,
        "timestamp": int(time.time()),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return {"saved": True, "label": label, "file": fname}


@app.post("/build-templates")
def build_templates():
    """
    Lee templates_data/A/*.json ... y saca el promedio de features por letra.
    Guarda templates.json y lo recarga en memoria.
    """
    templates = {}
    counts = {}

    for c in range(ord("A"), ord("Z") + 1):
        letter = chr(c)
        letter_dir = os.path.join(TEMPLATES_DIR, letter)
        if not os.path.isdir(letter_dir):
            continue

        feats = []
        for name in os.listdir(letter_dir):
            if not name.endswith(".json"):
                continue
            p = os.path.join(letter_dir, name)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                feats.append(data["features"])
            except:
                continue

        if len(feats) == 0:
            continue

        X = np.array(feats, dtype=np.float32)
        mean = X.mean(axis=0).tolist()

        templates[letter] = mean
        counts[letter] = int(len(feats))

    with open(TEMPLATES_OUT, "w", encoding="utf-8") as f:
        json.dump({"templates": templates, "counts": counts}, f, indent=2)

    load_templates_if_exist()

    return {"ok": True, "saved_to": "templates.json", "counts": counts}


@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Analiza un frame:
    - detecta mano con MediaPipe
    - extrae features normalizados
    - si existe templates.json, predice por template matching
    """
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    height, width, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)

    hands_output = []
    recognized_sign = None
    best_score = None

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_landmarks.landmark]

        handedness_label = None
        handedness_score = None
        if result.multi_handedness and len(result.multi_handedness) > 0:
            handedness_label = result.multi_handedness[0].classification[0].label
            handedness_score = float(result.multi_handedness[0].classification[0].score)

        hands_output.append({
            "handedness": handedness_label,
            "score": handedness_score,
            "landmarks": lm_list,
        })

        # features + predicción
        nlm = normalize_landmarks_xy(lm_list)
        feat = features_from_norm_lm(nlm)

        recognized_sign, best_score = predict_from_templates(feat)

    return JSONResponse({
        "image_width": width,
        "image_height": height,
        "hands": hands_output,
        "recognized_sign": recognized_sign,   # letra o null
        "best_score": best_score,             # mientras más pequeño mejor match
        "templates_loaded": bool(templates_cache),
        "match_threshold": MATCH_THRESHOLD,
    })


if __name__ == "__main__":
    # Forma recomendada
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
