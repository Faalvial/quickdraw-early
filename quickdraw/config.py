"""Configuración única del proyecto.

Todos los valores que el grupo podría querer cambiar están aquí, para no tener
números sueltos repartidos por el código. Ver docs/00_como_ejecutar.md.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

# 30 categorías del proyecto: 5 pares confundibles + 20 al azar (semilla 2026).
# Trompeta salió por tener mediana de 4 trazos y entró langosta, la siguiente
# categoría que sale con la misma semilla.
PARES = [("cat", "dog"), ("lion", "tiger"), ("horse", "zebra"),
         ("violin", "cello"), ("bear", "panda")]
CATEGORIAS_PARES = [c for par in PARES for c in par]
CATEGORIAS_AZAR = ["The Mona Lisa", "alarm clock", "bird", "brain", "bridge", "bus",
                   "calculator", "fireplace", "frog", "jail", "map", "remote control",
                   "scorpion", "sink", "stop sign", "sun", "teddy-bear", "lobster",
                   "whale", "yoga"]
CATEGORIAS = CATEGORIAS_PARES + CATEGORIAS_AZAR

# Fracciones del dibujo en las que se mide la exactitud: 10 %, 20 %, ..., 100 %.
FRACCIONES = tuple(round(0.1 * i, 1) for i in range(1, 11))


@dataclass(frozen=True)
class Config:
    # --- Rutas ---------------------------------------------------------------
    ruta_muestra: Path = Path("datos/muestra_30")   # archivos <categoria>.ndjson.gz
    ruta_trabajo: Path = Path("trabajo")            # cachés, checkpoints y resultados

    # --- Datos ---------------------------------------------------------------
    dibujos_por_categoria: int = 10_000      # reconocidos que se leen de cada archivo
    proporciones: tuple = (0.70, 0.15, 0.15)  # entrenamiento, validación, prueba
    semilla_particion: int = 2026
    min_trazos_progresivo: int = 3           # curvas y H5 solo con dibujos de K >= 3

    # --- Rasterización (ver docs/02_prefijos_y_rasterizacion.md) ------------
    resolucion: int = 28      # R: lado de la imagen final
    lienzo: int = 256         # se dibuja en grande y se reduce, para suavizar bordes
    grosor: int = 16          # grosor de línea en el lienzo (~1,75 px a 28x28)
    margen: int = 1           # m: margen en píxeles de la imagen final

    # --- Entrenamiento --------------------------------------------------------
    lote: int = 256
    epocas_max: int = 20
    paciencia: int = 3                 # épocas sin mejorar antes de parar
    semillas: tuple = (0, 1, 2)        # una por repetición de M2 y M2-P
    semillas_m1: tuple = (0,)          # M1 con una semilla; (0, 1, 2) si sobra tiempo
    dibujos_val_auc: int = 3000        # submuestra de validación para la AUC por época
    # Esquema de M2-P: "prefijos" (un prefijo al azar, el del plan) o "mixto"
    # (con probabilidad m2p_prob_completo, el dibujo completo). Ver docs/04.
    esquema_m2p: str = "prefijos"
    m2p_prob_completo: float = 0.5

    # --- Búsqueda de hiperparámetros (solo M2) --------------------------------
    busqueda_optimizadores: tuple = ("adam", "sgd")
    busqueda_lr: tuple = (3e-3, 1e-3, 3e-4)
    busqueda_dropout: tuple = (0.0, 0.3)
    busqueda_epocas: int = 5
    busqueda_fraccion_datos: float = 0.30
    # M1 solo prueba las tasas de aprendizaje con Adam y su dropout fijo de 0,3.
    m1_dropout: float = 0.3

    # --- Estadística ------------------------------------------------------------
    bootstrap_remuestreos: int = 1000
    semilla_bootstrap: int = 2026
    vecinos_knn: int = 10
    tsne_por_categoria: int = 150
    tsne_perplejidad: int = 30

    # --- Prueba ---------------------------------------------------------------
    # El conjunto de prueba se usa UNA sola vez, en la semana 13. Mientras esto
    # sea False, ninguna función del proyecto toca los dibujos de prueba.
    usar_prueba: bool = False

    @property
    def ruta_cache(self) -> Path:
        return Path(self.ruta_trabajo) / "cache"

    @property
    def ruta_modelos(self) -> Path:
        return Path(self.ruta_trabajo) / "modelos"

    @property
    def ruta_resultados(self) -> Path:
        return Path(self.ruta_trabajo) / "resultados"


def config_rapida(base: Config) -> Config:
    """Versión pequeña para comprobar en minutos que todo corre de punta a punta.

    Sus números no sirven para el paper: solo verifican que el código funciona.
    """
    return replace(base, dibujos_por_categoria=120, epocas_max=2, paciencia=1,
                   semillas=(0, 1, 2), dibujos_val_auc=300,
                   busqueda_optimizadores=("adam", "sgd"), busqueda_lr=(1e-3,),
                   busqueda_dropout=(0.0, 0.3), busqueda_epocas=1,
                   busqueda_fraccion_datos=0.5, bootstrap_remuestreos=200,
                   tsne_por_categoria=20)

# Nombres en español para tablas y figuras.
NOMBRES_ES = {
    "cat": "gato", "dog": "perro", "lion": "león", "tiger": "tigre", "horse": "caballo",
    "zebra": "cebra", "violin": "violín", "cello": "chelo", "bear": "oso", "panda": "panda",
    "The Mona Lisa": "La Mona Lisa", "alarm clock": "despertador", "bird": "pájaro",
    "brain": "cerebro", "bridge": "puente", "bus": "autobús", "calculator": "calculadora",
    "fireplace": "chimenea", "frog": "rana", "jail": "cárcel", "map": "mapa",
    "remote control": "control remoto", "scorpion": "escorpión", "sink": "lavabo",
    "stop sign": "señal de pare", "sun": "sol", "teddy-bear": "oso de peluche",
    "lobster": "langosta", "whale": "ballena", "yoga": "yoga"}
