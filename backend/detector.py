import torch
# Configuración de Seguridad Obligatoria para PyTorch 2.6+
# Le indicamos explícitamente al serializador que confíe en las clases internas de YOLOv8
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

from ultralytics import YOLO
import cv2

class DetectorAnimal:
    """
    Capa de Inteligencia Artificial: Se encarga de procesar los frames de video
    e identificar animales utilizando el modelo preentrenado YOLOv8.
    """
    def __init__(self, ruta_modelo: str = "modelo/yolov8n.pt"):
        """
        Inicializa la configuración del detector.
        """
        self.ruta_modelo = ruta_modelo
        self.modelo = None
        # Mapeo de IDs de clases de COCO Dataset para los 7 animales requeridos
        self.clases_animales = {
            15: "gato",
            16: "perro",
            17: "caballo",
            18: "oveja",
            19: "vaca",
            20: "elefante",
            21: "oso"
        }

    def cargar_modelo(self) -> bool:
        """
        Carga el modelo de YOLOv8 de manera segura en memoria.
        :return: True si se cargó correctamente, False en caso de error.
        """
        try:
            print(f"[INFO] Cargando modelo de Inteligencia Artificial desde '{self.ruta_modelo}'...")
            self.modelo = YOLO(self.ruta_modelo)
            return True
        except Exception as e:
            print(f"[EXCEPCIÓN] Error crítico al cargar el modelo YOLOv8: {e}")
            return False

    def detectar(self, frame, confianza_minima: float = 0.70):
        """
        Procesa un frame y busca los 7 animales definidos con un filtro de confianza estricto.
        
        :param frame: Matriz de imagen (BGR) proveniente de OpenCV.
        :param confianza_minima: Umbral estricto fijado en 70% (0.70).
        :return: Tupla con (animal_detectado, confianza, frame_dibujado) o (None, 0.0, frame)
        """
        if self.modelo is None:
            return None, 0.0, frame

        # Ejecutamos la inferencia de forma silenciosa (verbose=False)
        resultados = self.modelo(frame, verbose=False)[0]
        
        animal_detectado = None
        confianza_detectada = 0.0
        frame_resultado = frame.copy()

        # Iterar sobre las detecciones encontradas en el cuadro
        for box in resultados.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 1. Filtro de clase: Verificar si es uno de nuestros 7 animales
            if cls_id in self.clases_animales:
                # 2. Filtro estricto de negocio: Confianza mayor o igual al 70%
                if conf >= confianza_minima:
                    animal_detectado = self.clases_animales[cls_id]
                    confianza_detectada = conf
                    
                    # Obtener coordenadas del recuadro
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Dibujar recuadro estético sobre el animal (Verde esmeralda: BGR -> 76, 175, 80)
                    cv2.rectangle(frame_resultado, (x1, y1), (x2, y2), (76, 175, 80), 3)
                    
                    # Añadir etiqueta de texto con fondo para legibilidad
                    etiqueta = f"{animal_detectado.upper()} {conf*100:.1f}%"
                    cv2.putText(frame_resultado, etiqueta, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (76, 175, 80), 2, cv2.LINE_AA)
                    
                    # Retornamos la primera detección válida que cumpla con los criterios
                    break 

        return animal_detectado, confianza_detectada, frame_resultado


# --- SCRIPT DE PRUEBA INTEGRADA (CAMARA + IA) ---
if __name__ == "__main__":
    from backend.camara import Camara

    print("[TEST] Iniciando prueba integrada: Cámara + Detector IA...")
    componente_camara = Camara()
    componente_detector = DetectorAnimal()

    if componente_camara.conectar() and componente_detector.cargar_modelo():
        print("[TEST] Todo listo. Coloca un animal frente a la cámara (o búscalo en tu celular).")
        print("[TEST] Presiona 'q' para salir.")
        
        while True:
            exito, frame_original = componente_camara.obtener_frame()
            if not exito:
                break
                
            # Procesar el frame con el detector de IA
            animal, conf, frame_procesado = componente_detector.detectar(frame_original, confianza_minima=0.70)
            
            if animal:
                print(f"[IA] Detectado: {animal} | Confianza: {conf*100:.2f}%")
                
            cv2.imshow("Prueba Tecnica - Detector AnimalVision AI", frame_procesado)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        componente_camara.liberar()
        cv2.destroyAllWindows()
    else:
        print("[TEST] Error al inicializar los componentes.")