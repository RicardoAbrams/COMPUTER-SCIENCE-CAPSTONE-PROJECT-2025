# -*- coding: utf-8 -*-

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import uvicorn
import numpy as np
import cv2

# 🔴 IMPORT CORRECTO EN WINDOWS
from mediapipe.python.solutions import hands as mp_hands

app = FastAPI(title="SignalSpeak Backend")

# CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hands_detector = None


@app.on_event("startup")
def startup():
    global hands_detector
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("✅ MediaPipe Hands initialized")


@app.on_event("shutdown")
def shutdown():
    global hands_detector
    if hands_detector:
        hands_detector.close()
        print("🛑 MediaPipe Hands closed")


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SignalSpeak Python backend active"
    }


@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    img_bytes = await file.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    height, width, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = hands_detector.process(image_rgb)

    hands_output = []
    recognized_sign = None

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            landmarks = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in hand_landmarks.landmark
            ]

            handedness = None
            score = None
            if result.multi_handedness and i < len(result.multi_handedness):
                handedness = result.multi_handedness[i].classification[0].label
                score = float(result.multi_handedness[i].classification[0].score)

            hands_output.append({
                "handedness": handedness,
                "score": score,
                "landmarks": landmarks,
            })

        # Placeholder (luego ML real)
        recognized_sign = "A"

    return {
        "image_width": width,
        "image_height": height,
        "hands": hands_output,
        "recognized_sign": recognized_sign,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
