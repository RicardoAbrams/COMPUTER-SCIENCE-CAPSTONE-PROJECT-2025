# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gemini_router import router as gemini_router



import os
import threading
import time
import traceback
from typing import Optional, Dict, List, Tuple

import numpy as np
import cv2
import mediapipe as mp
import uvicorn

# =========================
# CONFIG
# =========================
HOST = "127.0.0.1"
PORT = 8000


BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates_data")

# Tu vector actual: 21 landmarks * (x,y,z) = 63
EXPECTED_DIM = 63

# Umbral para evitar clasificar cuando está lejos (ajusta luego)
DIST_THRESHOLD = 0.35

# En modo collect: guardar un frame cada X ms (para no guardar 20/seg)
COLLECT_MIN_INTERVAL_SEC = 0.12

# Para performance: cuántos .npy máximo por label para centroid (0/None = todos)
# Recomendado: 1500-3000 para centroid (suficiente y rápido)
MAX_FILES_PER_LABEL_FOR_CENTROID = 100

# =========================
# APP
# =========================
app = FastAPI(title="SignalSpeak Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:7109", "http://localhost:7109", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gemini_router)
# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands_detector: Optional[mp_hands.Hands] = None

# =========================
# TEMPLATES (promedios por label)
# =========================
_templates_loaded = False
_templates_lock = threading.Lock()
_template_centroids: Dict[str, np.ndarray] = {}  # label -> vector promedio (63)
_template_counts: Dict[str, int] = {}            # label -> cuántos npy se usaron
_last_label: Optional[str] = None                # debounce simple (analyze)

# collect rate limiter
_last_collect_ts: float = 0.0

def _log(msg: str):
    print(f"[SignalSpeak] {msg}", flush=True)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_templates_from_folder():
    """
    Lee templates_data/<label>/*.npy y crea un centroid (promedio) por label.
    SOLO acepta npy con dimensión EXPECTED_DIM (63).
    Versión robusta: no muere si un archivo está corrupto y loggea errores.
    """
    global _templates_loaded, _template_centroids, _template_counts

    if _templates_loaded:
        return
  
    with _templates_lock:
        if _templates_loaded:
            return

        start = time.time()
        _log(f"Cargando templates desde: {TEMPLATES_DIR}")

        try:
            if not os.path.isdir(TEMPLATES_DIR):
                _log("❌ No existe templates_data/.")
                _templates_loaded = False
                return

            labels = sorted(
                [d for d in os.listdir(TEMPLATES_DIR) if os.path.isdir(os.path.join(TEMPLATES_DIR, d))]
            )

            if not labels:
                _log("❌ templates_data existe pero no tiene subcarpetas (labels).")
                _templates_loaded = False
                return

            centroids: Dict[str, np.ndarray] = {}
            counts: Dict[str, int] = {}
            loaded_total = 0
            ignored_labels = 0

            for label in labels:
                label_dir = os.path.join(TEMPLATES_DIR, label)

                # Lista npy
                try:
                    files = [f for f in os.listdir(label_dir) if f.lower().endswith(".npy")]
                except Exception as e:
                    _log(f"⚠️ {label}: no pude listar archivos: {e}")
                    ignored_labels += 1
                    continue

                if not files:
                    continue

                # Limitar archivos por label para centroid (performance)
                if MAX_FILES_PER_LABEL_FOR_CENTROID and len(files) > MAX_FILES_PER_LABEL_FOR_CENTROID:
                    files = files[:MAX_FILES_PER_LABEL_FOR_CENTROID]

                vecs_ok: List[np.ndarray] = []
                seen_sizes = set()
                bad_files = 0

                for fname in files:
                    fpath = os.path.join(label_dir, fname)
                    try:
                        arr = np.load(fpath, allow_pickle=False)
                        arr = np.array(arr).reshape(-1).astype(np.float32)
                        seen_sizes.add(int(arr.shape[0]))
                        # valida dim + finitos
                        if arr.shape[0] == EXPECTED_DIM and np.isfinite(arr).all():
                            vecs_ok.append(arr)
                        else:
                            bad_files += 1
                    except Exception:
                        bad_files += 1
                        continue

                if not vecs_ok:
                    _log(f"⚠️ {label}: no hay npy válidos de {EXPECTED_DIM}. sizes={sorted(seen_sizes)[:15]} bad={bad_files}")
                    ignored_labels += 1
                    continue

                stacked = np.stack(vecs_ok, axis=0)
                centroid = stacked.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

                centroids[label] = centroid
                counts[label] = len(vecs_ok)
                loaded_total += len(vecs_ok)

                _log(f"✓ {label}: usados={len(vecs_ok)} bad={bad_files} sizes_sample={sorted(seen_sizes)[:5]}")

            _template_centroids = centroids
            _template_counts = counts
            _templates_loaded = True  # ✅ marcamos loaded aunque sea parcial

            _log(f"✅ Templates listos: {len(centroids)} labels, {loaded_total} samples en {time.time()-start:.2f}s")
            _log(f"ℹ️ Ignored labels: {ignored_labels}")
            _log(f"Dimensión vector esperada: {EXPECTED_DIM}")

        except Exception as e:
            _templates_loaded = False
            _template_centroids = {}
            _template_counts = {}
            _log("❌ ERROR cargando templates:")
            _log(str(e))
            _log(traceback.format_exc())
            return

def _vectorize_landmarks(lm_list: list) -> np.ndarray:
    """
    Convierte landmarks (x,y,z) a un vector 63 floats y lo normaliza.
    Centra respecto a la muñeca (landmark 0) para estabilidad.
    """
    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in lm_list], dtype=np.float32)
    pts = pts - pts[0]  # centra en muñeca
    v = pts.reshape(-1)  # 63
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def recognize_with_centroids(lm_list: list) -> Tuple[Optional[str], Optional[float]]:
    """
    Compara el vector actual contra centroides usando distancia euclidiana
    (en vectores normalizados). Devuelve (label, dist) o (None, dist_best).
    """
    if not _templates_loaded or not _template_centroids:
        return None, None

    v = _vectorize_landmarks(lm_list)
    if v.shape[0] != EXPECTED_DIM:
        return None, None

    best_label = None
    best_dist = None

    for label, c in _template_centroids.items():
        d = float(np.linalg.norm(v - c))
        if best_dist is None or d < best_dist:
            best_dist = d
            best_label = label

    if best_dist is None or best_dist > DIST_THRESHOLD:
        return None, best_dist

    return best_label, best_dist

# =========================
# LIFECYCLE
# =========================
@app.on_event("startup")
def startup():
    global hands_detector
    _log("Startup: inicializando MediaPipe Hands...")
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _log("✅ MediaPipe Hands initialized")

    # ✅ Cargar sync para ver errores (debug)
    _log("Cargando templates (sync para debug)...")
    load_templates_from_folder()

@app.on_event("shutdown")
def shutdown():
    global hands_detector
    _log("Shutdown: cerrando MediaPipe...")
    if hands_detector:
        hands_detector.close()
        hands_detector = None
    _log("✅ Cerrado")

# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "SignalSpeak Python backend active", "port": PORT}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "templates_loaded": _templates_loaded,
        "labels": len(_template_centroids) if _templates_loaded else 0,
        "samples_by_label": _template_counts if _templates_loaded else {},
        "templates_dir": TEMPLATES_DIR,
        "expected_dim": EXPECTED_DIM,
        "dist_threshold": DIST_THRESHOLD,
        "max_files_per_label_for_centroid": MAX_FILES_PER_LABEL_FOR_CENTROID,
    }

@app.get("/debug-templates")
def debug_templates():
    info = {"templates_dir": TEMPLATES_DIR, "labels": {}}
    if not os.path.isdir(TEMPLATES_DIR):
        info["error"] = "TEMPLATES_DIR does not exist"
        return info

    for d in sorted(os.listdir(TEMPLATES_DIR)):
        p = os.path.join(TEMPLATES_DIR, d)
        if not os.path.isdir(p):
            continue
        files = [f for f in os.listdir(p) if f.lower().endswith(".npy")]
        info["labels"][d] = {"npy_files": len(files), "sample_file": files[0] if files else None}
    return info

@app.post("/reload-templates")
def reload_templates():
    global _templates_loaded, _template_centroids, _template_counts
    with _templates_lock:
        _templates_loaded = False
        _template_centroids = {}
        _template_counts = {}
    load_templates_from_folder()
    return {"status": "ok", "templates_loaded": _templates_loaded, "labels": len(_template_centroids)}

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Endpoint que tu camara.js llama en modo analyze.

    ✅ Cambios:
    - Siempre calcula el "best_guess" (la letra más cercana) aunque NO pase el umbral.
    - "recognized_sign" solo se publica cuando hay estabilidad (majority vote) y pasa umbral.
    - "hold" mantiene la última letra estable por unos ms para evitar parpadeo.
    """

    # ---- AJUSTES DE ESTABILIDAD (puedes afinarlos) ----
    from collections import deque

    # Creamos estos globals si no existen (para que puedas copiar/pegar solo este endpoint)
    global _templates_loaded, _template_centroids, _last_label
    global _guess_window, _last_emit_time, _last_emitted_label

    try:
        _guess_window
    except NameError:
        _guess_window = deque(maxlen=7)  # ventana de 7 frames

    try:
        _last_emit_time
    except NameError:
        _last_emit_time = 0.0

    try:
        _last_emitted_label
    except NameError:
        _last_emitted_label = None

    HOLD_MS = 700          # cuánto tiempo mantiene la letra "pegada"
    MIN_VOTES = 3         # cuántos votos de 7 para declarar estable
    MIN_WINDOW = 3         # mínimo de frames antes de votar

    # -----------------------------------------------

    if hands_detector is None:
        return JSONResponse({"error": "Hands detector not initialized"}, status_code=500)

    if not _templates_loaded or not _template_centroids:
        return JSONResponse(
            {
                "error": "Templates not loaded",
                "templates_loaded": _templates_loaded,
                "labels": len(_template_centroids),
            },
            status_code=200,
        )

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
    best_guess = None
    best_dist = None

    now = time.time()

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_landmarks.landmark]
        hands_output.append({"landmarks": lm_list})

        # ===== 1) calcular best_guess SIEMPRE (sin umbral) =====
        v = _vectorize_landmarks(lm_list)
        if v.shape[0] == EXPECTED_DIM:

            # buscar el centroid más cercano
            for label, c in _template_centroids.items():
                d = float(np.linalg.norm(v - c))
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_guess = label

            # ===== 2) Ventana de estabilidad =====
            if best_guess is not None:
                _guess_window.append(best_guess)

            stable_label = None
            if len(_guess_window) >= MIN_WINDOW and best_guess is not None:
                # cuenta votos en ventana
                counts = {}
                for g in _guess_window:
                    counts[g] = counts.get(g, 0) + 1

                stable_label = max(counts, key=counts.get)
                votes = counts[stable_label]

                # ===== 3) Confirmar solo si pasa umbral + votos suficientes =====
                if best_dist is not None and best_dist <= DIST_THRESHOLD and votes >= MIN_VOTES:
                    recognized_sign = stable_label
                    _last_emitted_label = stable_label
                    _last_emit_time = now

            # ===== 4) HOLD: si no confirmó en este frame, mantiene el último emitido =====
            if recognized_sign is None and _last_emitted_label is not None:
                if (now - _last_emit_time) * 1000.0 <= HOLD_MS:
                    recognized_sign = _last_emitted_label

    else:
        # si no hay mano, resetea ventana para que no se “pegue” cosas viejas
        _guess_window.clear()

    return JSONResponse(
        {
            "image_width": width,
            "image_height": height,
            "hands": hands_output,
            "recognized_sign": recognized_sign,  # ✅ estable (con hold)
            "best_guess": best_guess,            # ✅ siempre el más cercano
            "best_distance": best_dist,
            "templates_loaded": _templates_loaded,
            "expected_dim": EXPECTED_DIM,
            "dist_threshold": DIST_THRESHOLD,
        }
    )


@app.post("/collect-template")
async def collect_template(
    file: UploadFile = File(...),
    label: str = Query(..., min_length=1, max_length=1, description="A..Z"),
):
    """
    Endpoint que tu camara.js llama en modo collect:
    POST /collect-template?label=A  (multipart file=frame.jpg)
    Guarda un .npy con vector de 63.
    """
    global _last_collect_ts, _templates_loaded

    if hands_detector is None:
        return JSONResponse({"error": "Hands detector not initialized"}, status_code=500)

    label = label.strip().upper()
    if len(label) != 1 or not ("A" <= label <= "Z"):
        return JSONResponse({"error": "Invalid label (must be A..Z)"}, status_code=400)

    # rate limit para no guardar demasiado
    now = time.time()
    if now - _last_collect_ts < COLLECT_MIN_INTERVAL_SEC:
        return JSONResponse({"saved": False, "reason": "rate_limited"}, status_code=200)
    _last_collect_ts = now

    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)

    if not result.multi_hand_landmarks:
        return JSONResponse({"saved": False, "reason": "no_hand_detected"}, status_code=200)

    hand_landmarks = result.multi_hand_landmarks[0]
    lm_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_landmarks.landmark]
    v = _vectorize_landmarks(lm_list)

    if v.shape[0] != EXPECTED_DIM:
        return JSONResponse({"saved": False, "reason": "wrong_dim", "dim": int(v.shape[0])}, status_code=200)

    label_dir = os.path.join(TEMPLATES_DIR, label)
    _ensure_dir(label_dir)

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{label}_{ts}_{int((now - int(now))*1000):03d}.npy"
    fpath = os.path.join(label_dir, fname)
    np.save(fpath, v.astype(np.float32))

    # recargar (async) para no reiniciar
    with _templates_lock:
        _templates_loaded = False
        _template_centroids.clear()
        _template_counts.clear()
    threading.Thread(target=load_templates_from_folder, daemon=True).start()

    return JSONResponse({"saved": True, "label": label, "file": fname, "dim": EXPECTED_DIM}, status_code=200)

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
