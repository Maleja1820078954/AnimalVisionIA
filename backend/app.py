from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
import threading

# Importamos módulos (asegúrate de que todos estos estén en la carpeta 'backend')
# En backend/app.py
from backend.camara import Camara
from backend.detector import DetectorAnimal
from backend.sonidos import ControladorSonido

app = FastAPI(title="AnimalVision AI - API de Backend")

@app.get("/")
async def root():
    return FileResponse("../frontend/index.html") # Ajustado para subir un nivel desde backend/

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
camara = Camara(indice_camara=0)
detector = DetectorAnimal()
audio = ControladorSonido()

animales_sistema = ["gato", "perro", "caballo", "oveja", "vaca", "elefante", "oso"]
audio.cargar_libreria_audios(animales_sistema)
detector.cargar_modelo()

estado_sistema = {"activo": False, "ultimo_animal": "Ninguno", "confianza": 0.0, "tiempo_inicio": None}
control_audio = {"ultimo_animal_detectado": None, "tiempo_ultima_alerta": 0.0, "intervalo_cooldown": 4.0}

hilo_ia = None
frame_actual_bytes = b''
candado = threading.Lock()

def bucle_procesamiento_ia():
    global estado_sistema, control_audio, frame_actual_bytes
    while estado_sistema["activo"]:
        exito, frame_original = camara.obtener_frame()
        if not exito or frame_original is None:
            time.sleep(0.01)
            continue

        animal_actual, conf, frame_procesado = detector.detectar(frame_original, confianza_minima=0.70)
        tiempo_actual = time.time()

        if animal_actual:
            estado_sistema["ultimo_animal"] = animal_actual.upper()
            estado_sistema["confianza"] = round(conf * 100, 1)
            if (animal_actual != control_audio["ultimo_animal_detectado"]) or \
               (animal_actual == control_audio["ultimo_animal_detectado"] and (tiempo_actual - control_audio["tiempo_ultima_alerta"]) > control_audio["intervalo_cooldown"]):
                audio.reproducir(animal_actual)
                control_audio["ultimo_animal_detectado"] = animal_actual
                control_audio["tiempo_ultima_alerta"] = tiempo_actual
        else:
            control_audio["ultimo_animal_detectado"] = None

        _, buffer = cv2.imencode('.jpg', frame_procesado)
        with candado:
            frame_actual_bytes = buffer.tobytes()
        time.sleep(0.03)

def generador_feed_video():
    global estado_sistema, frame_actual_bytes
    while estado_sistema["activo"]:
        if frame_actual_bytes:
            with candado:
                bytes_a_enviar = frame_actual_bytes
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytes_a_enviar + b'\r\n')
        time.sleep(0.03)

@app.get("/api/encender")
def encender():
    global hilo_ia, estado_sistema
    if not estado_sistema["activo"]:
        try:
            # Si falla la conexión a cámara física, el try-except evita que el servidor muera
            camara_conectada = camara.conectar()
        except Exception as e:
            print(f"Error detectado: {e}")
            camara_conectada = False
            
        if not camara_conectada:
            return {"status": "error", "message": "Fallo de hardware: No se encontró cámara"}

        estado_sistema["activo"] = True
        estado_sistema["tiempo_inicio"] = time.time()
        hilo_ia = threading.Thread(target=bucle_procesamiento_ia, daemon=True)
        hilo_ia.start()
        return {"status": "success", "message": "Sistema iniciado"}
    return {"status": "info", "message": "El sistema ya estaba activo"}

@app.get("/api/apagar")
def apagar():
    global estado_sistema, hilo_ia, frame_actual_bytes
    if estado_sistema["activo"]:
        estado_sistema["activo"] = False 
        camara.liberar()
        estado_sistema.update({"ultimo_animal": "Ninguno", "confianza": 0.0, "tiempo_inicio": None})
        frame_actual_bytes = b''
        hilo_ia = None
        return {"status": "success", "message": "Sistema detenido"}
    return {"status": "info", "message": "El sistema ya estaba apagado"}

@app.get("/api/estado")
def obtener_estado():
    tiempo_actividad = "00:00:00"
    if estado_sistema["activo"] and estado_sistema["tiempo_inicio"]:
        segundos = int(time.time() - estado_sistema["tiempo_inicio"])
        tiempo_actividad = time.strftime('%H:%M:%S', time.gmtime(segundos))
    return {**estado_sistema, "tiempo_actividad": tiempo_actividad}

@app.get("/api/video_feed")
def video_feed():
    if estado_sistema["activo"]:
        return StreamingResponse(generador_feed_video(), media_type="multipart/x-mixed-replace; boundary=frame")
    return {"status": "error", "message": "Sistema apagado"}