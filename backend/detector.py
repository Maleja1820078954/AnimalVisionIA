import torch
import os
from ultralytics import YOLO

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
    Capa de Inteligencia Artificial. Recibe un frame y devuelve datos
    (no imagen dibujada) para que el frontend renderice el recuadro.
    """
    def __init__(self, ruta_modelo: str = "modelo/yolov8n.pt"):
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
        """
        Devuelve (animal_detectado, confianza, bbox) en vez de una imagen.
        bbox es [x1, y1, x2, y2] en píxeles del frame recibido, o None si no hay detección.
        """
        if self.modelo is None:
            return None, 0.0, None

        resultados = self.modelo(frame, verbose=False)[0]

        for box in resultados.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in self.clases_animales and conf >= confianza_minima:
                animal_detectado = self.clases_animales[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return animal_detectado, conf, [x1, y1, x2, y2]

        return None, 0.0, None