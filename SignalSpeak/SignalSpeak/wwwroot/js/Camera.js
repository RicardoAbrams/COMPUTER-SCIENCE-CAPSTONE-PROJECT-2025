let currentStream = null;
let isRunning = false;

let currentMode = "analyze"; // "analyze" | "collect"
let currentLabel = "A";      // "A".."Z"
let dotNetRefGlobal = null;
let videoElGlobal = null;

let lastRecognized = null;   // para no spamear UI

function ensureVideoElement(videoElement) {
    if (!videoElement) throw new Error("No se recibió referencia al <video>.");
    videoElement.autoplay = true;
    videoElement.playsInline = true;
    videoElement.muted = true;
    return videoElement;
}

async function captureFrameToBlob(videoEl, quality = 0.85) {
    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth || 1280;
    canvas.height = videoEl.videoHeight || 720;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", quality)
    );

    return blob;
}

function getEndpointUrl() {
    if (currentMode === "collect") {
        return `http://localhost:8000/collect-template?label=${encodeURIComponent(currentLabel)}`;
    }
    return "http://localhost:8000/analyze-frame";
}

// ✅ llamado desde Blazor
export function setMode(mode) {
    if (mode !== "analyze" && mode !== "collect") return;
    currentMode = mode;
    console.log("Mode set:", currentMode);

    // reset spam control
    lastRecognized = null;
}

// ✅ llamado desde Blazor
export function setLabel(label) {
    if (!label || typeof label !== "string") return;
    const up = label.trim().toUpperCase();
    if (up.length !== 1) return;
    if (up < "A" || up > "Z") return;

    currentLabel = up;
    console.log("Label set:", currentLabel);
}

// ✅ llamado desde Blazor: graba por X ms en modo collect
export async function recordForMs(ms = 2000) {
    if (!isRunning || !videoElGlobal) throw new Error("Camera is not running.");
    if (!dotNetRefGlobal) throw new Error("dotNetRef not set.");

    const prevMode = currentMode;
    currentMode = "collect";
    lastRecognized = null;

    await dotNetRefGlobal.invokeMethodAsync(
        "UpdateStatus",
        `Recording label ${currentLabel} for ${ms} ms...`
    );

    await new Promise((resolve) => setTimeout(resolve, ms));

    currentMode = prevMode;

    await dotNetRefGlobal.invokeMethodAsync(
        "UpdateStatus",
        `Recording done for ${currentLabel}.`
    );
}

// ✅ llamado desde Blazor (tu función actual)
export async function startCamera(videoElement, dotNetRef) {
    console.log("startCamera called");

    const videoEl = ensureVideoElement(videoElement);

    // guardar referencias para recordForMs()
    dotNetRefGlobal = dotNetRef;
    videoElGlobal = videoEl;

    // Detener si ya había stream
    if (currentStream) {
        currentStream.getTracks().forEach((t) => t.stop());
        currentStream = null;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Este navegador no soporta getUserMedia().");
    }

    currentStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
    });

    videoEl.srcObject = currentStream;
    await videoEl.play();

    isRunning = true;

    if (dotNetRef) {
        await dotNetRef.invokeMethodAsync(
            "UpdateStatus",
            "Camera active - sending frames to Python backend..."
        );
    }

    // Loop de envío de frames
    const loop = async () => {
        if (!isRunning) return;

        try {
            const blob = await captureFrameToBlob(videoEl, 0.85);

            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");

            const url = getEndpointUrl();

            const response = await fetch(url, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                console.error("Backend error:", response.status, response.statusText);
                if (dotNetRef) {
                    await dotNetRef.invokeMethodAsync(
                        "UpdateStatus",
                        `Backend error: ${response.status}`
                    );
                }
            } else {
                const data = await response.json();

                if (dotNetRef) {
                    // En modo collect: el backend devuelve {saved:true/false,...}
                    if (currentMode === "collect") {
                        // opcional: mostrar saved si quieres
                        // (no spamear demasiado)
                    } else {
                        // modo analyze: recognized_sign puede ser null
                        const sign = data.recognized_sign ?? null;

                        // no spamear UI si no cambia
                        if (sign !== lastRecognized) {
                            lastRecognized = sign;
                            await dotNetRef.invokeMethodAsync(
                                "UpdateRecognizedText",
                                sign
                            );
                        }
                    }
                }
            }
        } catch (err) {
            console.error("Error sending frame:", err);
            if (dotNetRef) {
                await dotNetRef.invokeMethodAsync(
                    "UpdateStatus",
                    "Error sending frame (check backend/CORS)."
                );
            }
        }

        setTimeout(loop, 200);
    };

    loop();
}

// ✅ llamado desde Blazor (tu función actual)
export async function stopCamera(videoElement, dotNetRef) {
    console.log("stopCamera called");
    isRunning = false;

    try {
        if (currentStream) {
            currentStream.getTracks().forEach((t) => t.stop());
            currentStream = null;
        }

        if (videoElement) {
            videoElement.srcObject = null;
        }

        // reset globals
        dotNetRefGlobal = null;
        videoElGlobal = null;
        lastRecognized = null;

        if (dotNetRef) {
            await dotNetRef.invokeMethodAsync("UpdateStatus", "Camera stopped.");
            await dotNetRef.invokeMethodAsync("UpdateRecognizedText", null);
        }
    } catch (err) {
        console.error("stopCamera error:", err);
    }
}
