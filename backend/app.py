from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
import threading

# Importamos tus módulos de infraestructura existentes
from camara import Camara
from detector import DetectorAnimal
from sonidos import ControladorSonido

app = FastAPI(title="AnimalVision AI - API de Backend")

# Permitir que el Frontend web se conecte sin bloqueos de seguridad (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes globales del sistema
camara = Camara(indice_camara=0)
detector = DetectorAnimal()
audio = ControladorSonido()

animales_sistema = ["gato", "perro", "caballo", "oveja", "vaca", "elefante", "oso"]
audio.cargar_libreria_audios(animales_sistema)
detector.cargar_modelo()

# Variables globales compartidas de estado
estado_sistema = {
    "activo": False,
    "ultimo_animal": "Ninguno",
    "confianza": 0.0,
    "tiempo_inicio": None
}

# Variables de control para las alertas de sonido
control_audio = {
    "ultimo_animal_detectado": None,
    "tiempo_ultima_alerta": 0.0,
    "intervalo_cooldown": 4.0
}

# --- NUEVAS VARIABLES PARA SEPARAR EL FLUJO CORRECTAMENTE ---
hilo_ia = None
frame_actual_bytes = b''  # Guardará el último JPG procesado para el streaming
candado = threading.Lock()  # Evita que se lea el frame mientras se está escribiendo

def bucle_procesamiento_ia():
    """ ÚNICO hilo central que lee la cámara, procesa la IA y maneja el audio """
    global estado_sistema, control_audio, frame_actual_bytes
    
    while estado_sistema["activo"]:
        exito, frame_original = camara.obtener_frame()
        if not exito or frame_original is None:
            time.sleep(0.01)
            continue

        # Inferencia de YOLOv8 una única vez por frame
        animal_actual, conf, frame_procesado = detector.detectar(frame_original, confianza_minima=0.70)
        tiempo_actual = time.time()

        if animal_actual:
            estado_sistema["ultimo_animal"] = animal_actual.upper()
            estado_sistema["confianza"] = round(conf * 100, 1)

            # Control de reproducción de sonido corregido
            if (animal_actual != control_audio["ultimo_animal_detectado"]) or \
               (animal_actual == control_audio["ultimo_animal_detectado"] and \
                (tiempo_actual - control_audio["tiempo_ultima_alerta"]) > control_audio["intervalo_cooldown"]):
                
                audio.reproducir(animal_actual)
                control_audio["ultimo_animal_detectado"] = animal_actual
                control_audio["tiempo_ultima_alerta"] = tiempo_actual
        else:
            control_audio["ultimo_animal_detectado"] = None

        # Codificar el frame procesado en formato JPG
        _, buffer = cv2.imencode('.jpg', frame_procesado)
        
        # Guardar de manera segura en la variable global compartida
        with candado:
            frame_actual_bytes = buffer.tobytes()
        
        time.sleep(0.03) # Frecuencia controlada estable (~30 FPS)

def generador_feed_video():
    """ Endpoint pasivo: Solo toma la imagen global ya procesada y la transmite """
    global estado_sistema, frame_actual_bytes
    while estado_sistema["activo"]:
        if frame_actual_bytes:
            with candado:
                bytes_a_enviar = frame_actual_bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytes_a_enviar + b'\r\n')
        time.sleep(0.03)

# --- RECURSOS / ENDPOINTS DE LA API ---

@app.get("/api/encender")
def encender():
    """ Endpoint para activar el hardware y disparar el hilo de IA en segundo plano """
    global hilo_ia, estado_sistema
    if not estado_sistema["activo"]:
        if camara.conectar():
            estado_sistema["activo"] = True
            estado_sistema["tiempo_inicio"] = time.time()
            
            # Disparar el hilo único de fondo
            hilo_ia = threading.Thread(target=bucle_procesamiento_ia, daemon=True)
            hilo_ia.start()
            
            return {"status": "success", "message": "Sistema e hilo de IA inicializados"}
        return {"status": "error", "message": "Fallo de hardware de cámara"}
    return {"status": "info", "message": "El sistema ya estaba activo"}

@app.get("/api/apagar")
def apagar():
    """ Endpoint para apagar el sistema, detener el hilo y liberar recursos """
    global estado_sistema, hilo_ia, frame_actual_bytes
    if estado_sistema["activo"]:
        estado_sistema["activo"] = False 
        camara.liberar()
        estado_sistema["ultimo_animal"] = "Ninguno"
        estado_sistema["confianza"] = 0.0
        estado_sistema["tiempo_inicio"] = None
        frame_actual_bytes = b''
        hilo_ia = None
        return {"status": "success", "message": "Sistema y procesamiento detenidos"}
    return {"status": "info", "message": "El sistema ya estaba apagado"}

@app.get("/api/estado")
def obtener_estado():
    """ Devuelve métricas clave (KPIs) en formato JSON al Frontend """
    tiempo_actividad = "00:00:00"
    if estado_sistema["activo"] and estado_sistema["tiempo_inicio"]:
        segundos = int(time.time() - estado_sistema["tiempo_inicio"])
        tiempo_actividad = time.strftime('%H:%M:%S', time.gmtime(segundos))

    return {
        "activo": estado_sistema["activo"],
        "ultimo_animal": estado_sistema["ultimo_animal"],
        "confianza": estado_sistema["confianza"],
        "tiempo_actividad": tiempo_actividad
    }

@app.get("/api/video_feed")
def video_feed():
    """ Transmite el flujo continuo consumiendo el nuevo generador pasivo """
    if estado_sistema["activo"]:
        return StreamingResponse(generador_feed_video(), media_type="multipart/x-mixed-replace; boundary=frame")
    return {"status": "error", "message": "El sistema está apagado. No hay transmisión."}