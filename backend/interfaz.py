import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import time
import threading
import queue  # Estructura de datos segura para separar Front de Back

# Importación de componentes de infraestructura
from backend.camara import Camara
from backend.detector import DetectorAnimal
from backend.sonidos import ControladorSonido


class InterfazAnimalVision:
    """
    ===================================================================
    FRONTEND: Capa de Presentación (Solo gestiona UI y pinta datos)
    ===================================================================
    """
    def __init__(self, ventana_principal):
        self.root = ventana_principal
        self.root.title("AnimalVision AI - Dashboard de Monitoreo")
        self.root.geometry("1050x700")
        self.root.configure(bg="#121212") 

        # Instancias de Backend
        self.camara = Camara(indice_camara=0)
        self.detector = DetectorAnimal()
        self.audio = ControladorSonido()
        
        self.animales_sistema = ["gato", "perro", "caballo", "oveja", "vaca", "elefante", "oso"]
        self.audio.cargar_libreria_audios(self.animales_sistema)
        self.detector.cargar_modelo()

        # Variables de Control de Estado de la UI
        self.sistema_activo = False
        
        # COLA DE COMUNICACIÓN: Aquí el Backend deposita y el Frontend extrae
        self.cola_datos = queue.Queue(maxsize=2) 

        self.crear_interfaz_moderna()

    def crear_interfaz_moderna(self):
        # --- BARRA SUPERIOR (HEADER) ---
        header = tk.Frame(self.root, bg="#1E1E1E", height=70, bd=0, relief=tk.FLAT)
        header.pack(fill=tk.X, side=tk.TOP)
        
        titulo_lbl = tk.Label(header, text="ANIMALVISION AI", font=("Segoe UI", 20, "bold"), fg="#4CAF50", bg="#1E1E1E")
        titulo_lbl.pack(side=tk.LEFT, padx=30, pady=15)
        
        subtitulo_lbl = tk.Label(header, text="Monitoreo de Fauna en Tiempo Real", font=("Segoe UI", 10, "italic"), fg="#888888", bg="#1E1E1E")
        subtitulo_lbl.pack(side=tk.LEFT, pady=25)

        # --- CONTENEDOR PRINCIPAL DIVIDIDO (BODY) ---
        cuerpo_principal = tk.Frame(self.root, bg="#121212")
        cuerpo_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # LADO IZQUIERDO: Pantalla de Video
        contenedor_video = tk.Frame(cuerpo_principal, bg="#1E1E1E", bd=1, relief=tk.FLAT)
        contenedor_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.pantalla_lbl = tk.Label(contenedor_video, bg="#151515", text="SISTEMA FUERA DE LÍNEA\n\nPresione 'Iniciar Monitoreo' para encender la IA", font=("Segoe UI", 13), fg="#666666")
        self.pantalla_lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LADO DERECHO: Panel de Control y Estado
        panel_derecho = tk.Frame(cuerpo_principal, bg="#1E1E1E", width=320)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        panel_derecho.pack_propagate(False)

        # Tarjeta Interna 1: Controles de Hardware
        lbl_seccion1 = tk.Label(panel_derecho, text="CONTROLES DE SISTEMA", font=("Segoe UI", 11, "bold"), fg="#FFFFFF", bg="#1E1E1E")
        lbl_seccion1.pack(anchor=tk.W, padx=20, pady=(20, 10))

        self.btn_iniciar = tk.Button(panel_derecho, text="▶ Iniciar Monitoreo", font=("Segoe UI", 11, "bold"), 
                                      bg="#4CAF50", fg="white", activebackground="#45a049", activeforeground="white",
                                      width=24, height=2, bd=0, cursor="hand2", command=self.encender_sistema)
        self.btn_iniciar.pack(pady=8, padx=20)

        self.btn_detener = tk.Button(panel_derecho, text="■ Detener Sistema", font=("Segoe UI", 11, "bold"), 
                                      bg="#E53935", fg="white", activebackground="#d32f2f", activeforeground="white",
                                      width=24, height=2, bd=0, cursor="hand2", command=self.apagar_sistema, state=tk.DISABLED)
        self.btn_detener.pack(pady=8, padx=20)

        # Separador estético
        separador = tk.Frame(panel_derecho, bg="#333333", height=1)
        separador.pack(fill=tk.X, padx=20, pady=20)

        # Tarjeta Interna 2: Lista de Objetivos de la IA
        lbl_seccion2 = tk.Label(panel_derecho, text="ESPECIES OBJETIVO (YOLOv8)", font=("Segoe UI", 11, "bold"), fg="#888888", bg="#1E1E1E")
        lbl_seccion2.pack(anchor=tk.W, padx=20, pady=(0, 10))

        lista_frame = tk.Frame(panel_derecho, bg="#151515", bd=1)
        lista_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        texto_especies = " ✔ Perro          ✔ Gato\n\n ✔ Vaca           ✔ Oveja\n\n ✔ Caballo       ✔ Elefante\n\n ✔ Oso"
        lbl_especies = tk.Label(lista_frame, text=texto_especies, font=("Segoe UI", 10), fg="#AAAAAA", bg="#151515", justify=tk.LEFT)
        lbl_especies.pack(pady=20, padx=15, anchor=tk.W)

        # --- BARRA DE ESTADO INFERIOR (FOOTER) ---
        self.footer = tk.Frame(self.root, bg="#1E1E1E", height=35)
        self.footer.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_lbl = tk.Label(self.footer, text=" Estado: Listo para conectar hardware", font=("Segoe UI", 10), fg="#888888", bg="#1E1E1E", anchor=tk.W)
        self.status_lbl.pack(fill=tk.X, side=tk.LEFT, padx=15, pady=8)

    def encender_sistema(self):
        """ Arranca el Backend y la escucha del Frontend """
        if not self.sistema_activo:
            if self.camara.conectar():
                self.sistema_activo = True
                self.btn_iniciar.config(state=tk.DISABLED, bg="#334d38", fg="#777777")
                self.btn_detener.config(state=tk.NORMAL, bg="#E53935", fg="white")
                self.status_lbl.config(text=" Estado: Monitoreando transmisión de video...", fg="#4CAF50")
                
                # Lanzar el Backend en un hilo limpio aislado
                hilo_backend = threading.Thread(target=bucle_procesamiento_backend, args=(self,), daemon=True)
                hilo_backend.start()
                
                # Lanzar la escucha asíncrona del Frontend
                self.escuchar_cola_backend()

    def apagar_sistema(self):
        """ Apaga la UI y detiene el flujo """
        if self.sistema_activo:
            self.sistema_activo = False
            self.camara.liberar()
            
            self.btn_iniciar.config(state=tk.NORMAL, bg="#4CAF50", fg="white")
            self.btn_detener.config(state=tk.DISABLED, bg="#4A1C1A", fg="#777777")
            self.pantalla_lbl.config(image="", text="SISTEMA FUERA DE LÍNEA\n\nPresione 'Iniciar Monitoreo' para encender la IA", font=("Segoe UI", 13), fg="#666666")
            self.status_lbl.config(text=" Estado: Monitoreo desactivado de forma segura.", fg="#888888")

    def escuchar_cola_backend(self):
        """ El Frontend revisa periódicamente la cola sin trancar los hilos """
        if self.sistema_activo:
            try:
                # Intenta sacar un frame procesado de la cola de forma inmediata
                imagen_tk, texto_estado = self.cola_datos.get_nowait()
                
                # Actualizar elementos visuales (Frontend Puro)
                self.pantalla_lbl.config(image=imagen_tk)
                self.pantalla_lbl.image = imagen_tk
                if texto_estado:
                    self.status_lbl.config(text=texto_estado, fg="#4CAF50")
            except queue.Empty:
                pass # Si la cola está vacía, no hace nada y continúa
            
            # Volver a chequear en 20 milisegundos
            self.root.after(20, self.escuchar_cola_backend)


"""
===================================================================
BACKEND: Motor de Procesamiento Aislado (No toca componentes de UI)
===================================================================
"""
def bucle_procesamiento_backend(app):
    ultimo_animal_detectado = None
    tiempo_ultima_alerta = 0.0
    intervalo_cooldown = 4.0

    while app.sistema_activo:
        try:
            exito, frame_original = app.camara.obtener_frame()
            if not exito or frame_original is None:
                time.sleep(0.01)
                continue

            # 1. Ejecutar inferencia de YOLOv8
            animal_actual, confianza, frame_procesado = app.detector.detectar(frame_original, confianza_minima=0.70)
            
            tiempo_actual = time.time()
            texto_estado_actual = ""
            
            # 2. Controlar Alertas de Audio
            if animal_actual:
                if (animal_actual != ultimo_animal_detectado) or \
                   (animal_actual == ultimo_animal_detectado and (tiempo_actual - tiempo_ultima_alerta) > intervalo_cooldown):
                    
                    texto_estado_actual = f" Estado - Alerta: {animal_actual.upper()} detectado ({confianza*100:.1f}%)"
                    app.audio.reproducir(animal_actual)
                    
                    ultimo_animal_detectado = animal_actual
                    tiempo_ultima_alerta = tiempo_actual
            else:
                ultimo_animal_detectado = None

            # 3. Formatear la matriz de OpenCV a Formato Pil/Tkinter
            frame_reducido = cv2.resize(frame_procesado, (640, 480))
            frame_rgb = cv2.cvtColor(frame_reducido, cv2.COLOR_BGR2RGB)
            imagen_pil = Image.fromarray(frame_rgb)
            imagen_tk = ImageTk.PhotoImage(image=imagen_pil)

            # 4. Enviar a la cola para el Frontend (Si está llena, saca el viejo y mete el nuevo)
            if app.cola_datos.full():
                try:
                    app.cola_datos.get_nowait()
                except queue.Empty:
                    pass
            app.cola_datos.put_nowait((imagen_tk, texto_estado_actual))

            # Evitar uso innecesario de CPU
            time.sleep(0.02)

        except Exception as e:
            print(f"[ERROR CRÍTICO EN BACKEND] {e}")
            break