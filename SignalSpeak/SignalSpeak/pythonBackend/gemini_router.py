# -*- coding: utf-8 -*-
import os
import uuid
import json
import time
import re
from typing import Any, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException
from google import genai

router = APIRouter(prefix="/gemini", tags=["gemini"])

MODEL_ID = "gemini-flash-latest"

BASE_DIR = os.path.dirname(__file__)
TMP_DIR = os.path.join(BASE_DIR, "tmp_uploads")
os.makedirs(TMP_DIR, exist_ok=True)

client = genai.Client()

# ✅ Prompt más estricto: no markdown / no ```json
SYSTEM_PROMPT = (
    "You are an expert sign language interpreter. "
    "Analyze the uploaded video and return the signed meaning. "
    "IMPORTANT: Return ONLY raw JSON. "
    "Do NOT use markdown. Do NOT use code fences like ```json. "
    "Return exactly this schema: "
    '{"transcription": string, "language": "en"|"es"|"unknown", "confidence": number, "notes": string}.'
)

# Ponlo False cuando ya esté estable
DEBUG_GEMINI = False


def _extract_state(obj) -> str:
    candidates = []
    candidates.append(getattr(obj, "state", None))
    file_obj = getattr(obj, "file", None)
    candidates.append(getattr(file_obj, "state", None))
    candidates.append(getattr(obj, "status", None))

    for c in candidates:
        if c is None:
            continue
        s = str(c).strip().upper()
        if "ACTIVE" in s:
            return "ACTIVE"
        if "PROCESS" in s:
            return "PROCESSING"
        if "FAILED" in s:
            return "FAILED"
        if "ERROR" in s:
            return "ERROR"
    return "UNKNOWN"


def _wait_for_file_active(file_name: str, timeout_sec: int = 180, poll_sec: float = 1.0):
    deadline = time.time() + timeout_sec
    last_state = "UNKNOWN"

    while time.time() < deadline:
        f = client.files.get(name=file_name)
        last_state = _extract_state(f)

        if DEBUG_GEMINI:
            print(f"[Gemini] file={file_name} state={last_state}", flush=True)

        if last_state == "ACTIVE":
            return f

        if last_state in ("FAILED", "ERROR"):
            raise RuntimeError(f"Gemini file processing failed. state={last_state}")

        time.sleep(poll_sec)

    raise TimeoutError(f"Timed out waiting for file ACTIVE. last_state={last_state}")


def _extract_json_from_text(text: str) -> Dict[str, Any] | None:
    """
    Intenta parsear JSON directo.
    Si Gemini lo envía dentro de ```json ... ```, extrae el bloque.
    """
    # 1) JSON directo
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Extraer bloque ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) Extraer el primer {...} grande (fallback)
    m2 = re.search(r"(\{.*\})", text, re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def _normalize_confidence(value: Any) -> float:
    """
    Asegura 0..1.
    Si Gemini manda 5, lo convertimos a 0.5.
    """
    try:
        x = float(value)
        if x > 1.0:
            # ejemplo 5 -> 0.5, 7 -> 0.7, 10 -> 1.0
            x = x / 10.0
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        return x
    except Exception:
        return 0.5


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video (content-type video/*).")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY environment variable.")

    safe_name = (file.filename or "upload.mp4").replace(" ", "_")
    tmp_name = f"{uuid.uuid4()}_{safe_name}"
    tmp_path = os.path.join(TMP_DIR, tmp_name)

    data = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(data)

    try:
        uploaded = client.files.upload(file=tmp_path)

        uploaded_active = _wait_for_file_active(
            uploaded.name,
            timeout_sec=180,
            poll_sec=1.0,
        )

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[uploaded_active, SYSTEM_PROMPT],
        )

        text = (response.text or "").strip()
        if not text:
            return {
                "transcription": "",
                "language": "unknown",
                "confidence": 0.0,
                "notes": "Gemini returned empty response."
            }

        parsed = _extract_json_from_text(text)

        # ✅ si vino JSON válido, lo devolvemos limpio
        if parsed and "transcription" in parsed:
            parsed["language"] = str(parsed.get("language", "unknown"))
            parsed["confidence"] = _normalize_confidence(parsed.get("confidence", 0.5))
            parsed["notes"] = str(parsed.get("notes", ""))
            parsed["transcription"] = str(parsed.get("transcription", "")).strip()
            return parsed

        # fallback: texto crudo
        return {
            "transcription": text,
            "language": "unknown",
            "confidence": 0.5,
            "notes": "Model did not return strict JSON; returned raw text."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
