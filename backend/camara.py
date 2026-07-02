import cv2

class Camara:
    """
    Componente de Hardware: Gestiona la conexión, captura de frames
    y liberación segura de la cámara web utilizando OpenCV.
    """
    def __init__(self, indice_camara: int = 0):
        """
        Inicializa la configuración de la cámara.
        :param indice_camara: ID del dispositivo de video (0 para la webcam integrada).
        """
        self.indice_camara = indice_camara
        self.captura = None

    def conectar(self) -> bool:
        """
        Establece conexión con el hardware de la cámara web.
        :return: True si se conectó con éxito, False en caso contrario.
        """
        try:
            self.captura = cv2.VideoCapture(self.indice_camara)
            if not self.captura.isOpened():
                print(f"[ERROR] No se pudo acceder a la cámara en el índice {self.indice_camara}.")
                return False
            return True
        except Exception as e:
            print(f"[EXCEPCIÓN] Error inesperado al conectar la cámara: {e}")
            return False

    def obtener_frame(self):
        """
        Captura el cuadro (frame) actual de la transmisión.
        :return: Tupla (bool, frame) donde el booleano indica el éxito de la lectura.
        """
        if self.captura is not None and self.captura.isOpened():
            return self.captura.read()
        return False, None

    def liberar(self) -> None:
        """
        Libera el recurso de hardware para que otras aplicaciones puedan usarlo.
        """
        if self.captura is not None:
            self.captura.release()
            self.captura = None
            print("[INFO] Recurso de la cámara liberado correctamente.")


# --- PRUEBA UNITARIA AISLADA ---
if __name__ == "__main__":
    print("[TEST] Iniciando prueba de la clase Camara...")
    mi_camara = Camara(indice_camara=0)
    
    if mi_camara.conectar():
        print("[TEST] Cámara conectada. Presiona 'q' en la ventana de video para salir.")
        while True:
            exito, frame = mi_camara.obtener_frame()
            if not exito:
                break
                
            cv2.imshow("AnimalVision AI - Test Camara", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        mi_camara.liberar()
        cv2.destroyAllWindows()
        