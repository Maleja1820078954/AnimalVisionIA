import pygame
import os

class ControladorSonido:
    """
    Capa Multimedia: Gestiona la carga y reproducción de alertas de audio
    para cada animal utilizando Pygame Mixer en un canal dedicado.
    """
    def __init__(self, carpeta_sonidos: str = "sonidos"):
        """
        Inicializa el subsistema de audio de Pygame.
        """
        self.carpeta_sonidos = carpeta_sonidos
        self.sonidos_cargados = {}
        
        # Inicializar el mezclador de Pygame de forma segura
        try:
            pygame.mixer.init()
            print("[INFO] Subsistema de audio (Pygame Mixer) inicializado con éxito.")
        except Exception as e:
            print("[EXCEPCIÓN] Error al inicializar Pygame Mixer: {e}")

    def cargar_libreria_audios(self, lista_animales: list) -> None:
        """
        Busca y carga en memoria los archivos de audio para los animales configurados.
        Soporta extensiones .mp3 y .wav.
        
        :param lista_animales: Lista con los nombres de los animales a mapear.
        """
        print("[INFO] Cargando biblioteca de efectos de sonido...")
        for animal in lista_animales:
            archivo_mp3 = os.path.join(self.carpeta_sonidos, f"{animal}.mp3")
            archivo_wav = os.path.join(self.carpeta_sonidos, f"{animal}.wav")
            
            # Verificar cuál formato existe localmente
            ruta_valida = None
            if os.path.exists(archivo_mp3):
                ruta_valida = archivo_mp3
            elif os.path.exists(archivo_wav):
                ruta_valida = archivo_wav
                
            if ruta_valida:
                try:
                    self.sonidos_cargados[animal] = pygame.mixer.Sound(ruta_valida)
                    print(f"  -> Audio de [{animal}] cargado correctamente.")
                except Exception as e:
                    print(f"  [ERROR] No se pudo procesar el archivo {ruta_valida}: {e}")
            else:
                print(f"  [ADVERTENCIA] No se encontró archivo de audio para el animal: '{animal}'")

    def reproducir(self, animal: str) -> None:
        """
        Reproduce el sonido asociado al animal en un canal libre.
        No interrumpe el flujo principal del video.
        """
        if animal in self.sonidos_cargados:
            try:
                # Comprobar si el mezclador no está ocupado reproduciendo ya un sonido
                if not pygame.mixer.get_busy():
                    self.sonidos_cargados[animal].play()
                    print(f"[AUDIO] Reproduciendo efecto sonoro de: {animal.upper()}")
            except Exception as e:
                print(f"[EXCEPCIÓN] Error al reproducir audio de {animal}: {e}")
        else:
            print(f"[AUDIO] Intento de reproducción fallido: No hay audio cargado para '{animal}'")


# --- PRUEBA UNITARIA AISLADA ---
if __name__ == "__main__":
    import time
    print("[TEST] Iniciando prueba aislada del componente de sonido...")
    
    # Instanciar controlador
    controlador = ControladorSonido()
    
    # Forzar carga de prueba para los 7 animales requeridos
    animales_sistema = ["gato", "perro", "caballo", "oveja", "vaca", "elefante", "oso"]
    controlador.cargar_libreria_audios(animales_sistema)
    
    print("\n[TEST] Ejecutando simulación de reproducción (Sonará una 'vaca' si el archivo existe)...")
    controlador.reproducir("vaca")
    
    # Pequeña espera para dar tiempo a que suene en la prueba de consola
    time.sleep(3)
    print("[TEST] Fin de la prueba de audio.")