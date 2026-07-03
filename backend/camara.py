import cv2
import os

class Camara:
    """
    Capa de Hardware adaptable: Gestiona captura desde Webcam o archivo MP4
    dependiendo de si el sistema está en la nube o local.
    """
    def __init__(self, indice_camara: int = 0):
        self.indice_camara = indice_camara
        self.captura = None
        # Detectamos si estamos en un entorno de servidor (Render)
        self.es_entorno_nube = os.environ.get("RAILWAY_ENVIRONMENT") is not None

    def conectar(self) -> bool:
        """
        Intenta conectar a la cámara web o, si falla/nube, al video de prueba.
        """
        # Si estamos en la nube, forzamos la lectura del video de prueba
        if self.es_entorno_nube:
            ruta_video = "video_demo.mp4"
            print(f"[INFO] Entorno Nube detectado. Intentando abrir: {ruta_video}")
            self.captura = cv2.VideoCapture(ruta_video)
        else:
            # Local: probamos webcam
            self.captura = cv2.VideoCapture(self.indice_camara)
            
        if self.captura is None or not self.captura.isOpened():
            print(f"[ERROR] No se pudo acceder a la fuente de video.")
            return False
        return True

    def obtener_frame(self):
        if self.captura is not None and self.captura.isOpened():
            exito, frame = self.captura.read()
            
            # Si el video de prueba termina en la nube, lo reiniciamos automáticamente (Bucle infinito)
            if not exito and self.es_entorno_nube:
                self.captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.captura.read()
                
            return exito, frame
        return False, None

    def liberar(self) -> None:
        if self.captura is not None:
            self.captura.release()
            self.captura = None
            print("[INFO] Recurso de video liberado.")

# --- PRUEBA UNITARIA SEGURA PARA LA NUBE ---
if __name__ == "__main__":
    print("[TEST] Iniciando prueba... (No abrirá ventanas si detecta servidor)")
    mi_camara = Camara()
    
    if mi_camara.conectar():
        print("[TEST] Conexión establecida.")
        exito, frame = mi_camara.obtener_frame()
        if exito:
            print("[TEST] Frame capturado exitosamente.")
        mi_camara.liberar()