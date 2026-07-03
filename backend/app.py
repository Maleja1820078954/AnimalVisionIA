from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import os

from backend.detector import DetectorAnimal

app = FastAPI(title="AnimalVision AI - API de Backend")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
SONIDOS_DIR = os.path.join(BASE_DIR, "sonidos")

app.mount("/sonidos", StaticFiles(directory=SONIDOS_DIR), name="sonidos")

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el detector (ya no hay Camara ni ControladorSonido en el backend)
detector = DetectorAnimal()
detector.cargar_modelo()

animales_sistema = ["gato", "perro", "caballo", "oveja", "vaca", "elefante", "oso"]

# Historial simple en memoria para esta sesión (contador por animal)
historial = {animal: 0 for animal in animales_sistema}


@app.post("/api/detectar")
async def detectar(imagen: UploadFile = File(...)):
    """
    Recibe un frame capturado desde la cámara del navegador (getUserMedia),
    corre YOLOv8 sobre él y devuelve el resultado como JSON.
    No guarda ni reenvía la imagen: el frontend ya la tiene.
    """
    try:
        contenido = await imagen.read()
        arreglo = np.frombuffer(contenido, dtype=np.uint8)
        frame = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "error", "message": "No se pudo decodificar la imagen recibida"}

        animal, confianza, bbox = detector.detectar(frame, confianza_minima=0.70)

        if animal:
            historial[animal] += 1
            return {
                "status": "success",
                "animal": animal,
                "confianza": round(confianza * 100, 1),
                "bbox": bbox,  # [x1, y1, x2, y2] en píxeles del frame recibido
            }

        return {"status": "success", "animal": None, "confianza": 0.0, "bbox": None}

    except Exception as e:
        print(f"[ERROR] Fallo al procesar frame: {e}")
        return {"status": "error", "message": "Error interno al procesar la imagen"}


@app.get("/api/historial")
def obtener_historial():
    return {"status": "success", "historial": historial}


@app.post("/api/reiniciar_historial")
def reiniciar_historial():
    for animal in historial:
        historial[animal] = 0
    return {"status": "success", "message": "Historial reiniciado"}