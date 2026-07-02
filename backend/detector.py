import torch
import os
from ultralytics import YOLO
import cv2

# Configuración de Seguridad para PyTorch 2.6+
torch.serialization.add_safe_globals([
    "ultralytics.nn.tasks.DetectionModel",
    "ultralytics.nn.modules.block.C2f",
    "ultralytics.nn.modules.conv.Conv",
    "ultralytics.nn.modules.block.DFL",
    "ultralytics.nn.modules.block.Bottleneck",
    "ultralytics.nn.modules.head.Detect",
    "ultralytics.nn.modules.conv.Concat",
    "torch.nn.modules.upsampling.Upsample"
])

class DetectorAnimal:
    """
    Capa de Inteligencia Artificial optimizada para entornos Headless (Nube).
    """
    def __init__(self, ruta_modelo: str = "modelo/yolov8n.pt"):
        # Construimos una ruta absoluta basada en la ubicación de este archivo
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ruta_modelo = os.path.join(base_dir, ruta_modelo)
        self.modelo = None
        
        self.clases_animales = {
            15: "gato", 16: "perro", 17: "caballo", 
            18: "oveja", 19: "vaca", 20: "elefante", 21: "oso"
        }

    def cargar_modelo(self) -> bool:
        try:
            print(f"[INFO] Cargando modelo desde: {self.ruta_modelo}")
            if not os.path.exists(self.ruta_modelo):
                print(f"[ERROR] Archivo no encontrado en {self.ruta_modelo}")
                return False
            self.modelo = YOLO(self.ruta_modelo)
            return True
        except Exception as e:
            print(f"[EXCEPCIÓN] Error crítico al cargar YOLOv8: {e}")
            return False

    def detectar(self, frame, confianza_minima: float = 0.70):
        if self.modelo is None:
            return None, 0.0, frame

        # Inferencia silenciosa
        resultados = self.modelo(frame, verbose=False)[0]
        
        animal_detectado = None
        confianza_detectada = 0.0
        frame_resultado = frame.copy()

        for box in resultados.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in self.clases_animales and conf >= confianza_minima:
                animal_detectado = self.clases_animales[cls_id]
                confianza_detectada = conf
                
                # Coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Dibujar sin abrir ventanas
                cv2.rectangle(frame_resultado, (x1, y1), (x2, y2), (76, 175, 80), 3)
                etiqueta = f"{animal_detectado.upper()} {conf*100:.1f}%"
                cv2.putText(frame_resultado, etiqueta, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (76, 175, 80), 2, cv2.LINE_AA)
                break 

        return animal_detectado, confianza_detectada, frame_resultado

