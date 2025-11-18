// wwwroot/js/camera.js

let currentStream = null;

// Opcional: asegurar que el elemento video es válido
function ensureVideoElement(videoElement) {
    if (!videoElement) {
        throw new Error("No se recibió referencia al elemento <video>.");
    }

    videoElement.autoplay = true;
    videoElement.playsInline = true;
    videoElement.muted = true;

    return videoElement;
}

export async function startCamera(videoElementRef) {
    console.log("startCamera llamado con:", videoElementRef);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Este navegador no soporta getUserMedia.");
    }

    const videoElement = ensureVideoElement(videoElementRef);

    // Si había una cámara activa, detenla
    if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
        currentStream = null;
    }

    try {
        let stream;
        // Intento con constraints “bonitos”
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: "user"
                },
                audio: false
            });
        } catch (err) {
            console.warn("Fallo con constraints detallados, usando fallback sencillo:", err);
            stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });
        }

        console.log("Stream obtenido:", stream);

        currentStream = stream;
        videoElement.srcObject = stream;
        await videoElement.play();

        console.log("Video reproduciéndose");
        return true;
    } catch (err) {
        console.error("Error en getUserMedia:", err);
        throw err;
    }
}

export function stopCamera(videoElementRef) {
    console.log("stopCamera llamado con:", videoElementRef);
    const videoElement = videoElementRef ?? null;

    if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
        currentStream = null;
    }

    if (videoElement) {
        videoElement.srcObject = null;
    }
}

export function capturePhoto(videoElementRef) {
    console.log("capturePhoto llamado con:", videoElementRef);
    const videoElement = ensureVideoElement(videoElementRef);

    if (!videoElement.videoWidth || !videoElement.videoHeight) {
        throw new Error("Video no listo para captura.");
    }

    const canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoElement, 0, 0);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    console.log("Foto capturada, longitud dataURL:", dataUrl.length);

    return dataUrl;
}
