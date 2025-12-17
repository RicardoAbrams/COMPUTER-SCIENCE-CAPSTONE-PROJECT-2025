let currentStream = null;
let isRunning = false;

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

export async function startCamera(videoElement, dotNetRef) {
    console.log("startCamera called");

    const videoEl = ensureVideoElement(videoElement);

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
        await dotNetRef.invokeMethodAsync("UpdateStatus", "Camera active - sending frames to Python backend...");
    }

    // Loop de envío de frames
    const loop = async () => {
        if (!isRunning) return;

        try {
            const blob = await captureFrameToBlob(videoEl, 0.85);

            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");

            const response = await fetch("http://localhost:8000/analyze-frame", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                console.error("Backend error:", response.status, response.statusText);
                if (dotNetRef) {
                    await dotNetRef.invokeMethodAsync("UpdateStatus", `Backend error: ${response.status}`);
                }
            } else {
                const data = await response.json();
                console.log("Backend response:", data);

                if (dotNetRef) {
                    // recognized_sign puede venir null si no detecta mano
                    await dotNetRef.invokeMethodAsync("UpdateRecognizedText", data.recognized_sign);
                }
            }
        } catch (err) {
            console.error("Error sending frame:", err);
            if (dotNetRef) {
                await dotNetRef.invokeMethodAsync("UpdateStatus", "Error sending frame (check backend/CORS).");
            }
        }

        // Ajusta el rate (ms): 150–250 es ok para demo
        setTimeout(loop, 200);
    };

    loop();
}

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

        if (dotNetRef) {
            await dotNetRef.invokeMethodAsync("UpdateStatus", "Camera stopped.");
            await dotNetRef.invokeMethodAsync("UpdateRecognizedText", null);
        }
    } catch (err) {
        console.error("stopCamera error:", err);
    }
}
