"""Prefijos, normalización causal y rasterización.

Implementa las ecuaciones de la sección 3.2 del avance. Ver
docs/02_prefijos_y_rasterizacion.md.
"""
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import Config


def k_de_fraccion(K: int, f: float) -> int:
    """Número de trazos del prefijo para la fracción f: k = max(1, ceil(f*K)).

    Se resta un epsilon para que errores de redondeo (0,3*10 = 3,0000000004)
    no suban k en un trazo.
    """
    return max(1, math.ceil(f * K - 1e-9))


def caja(trazos):
    """Caja delimitadora (x_min, y_min, x_max, y_max) de una lista de trazos."""
    xs = [x for t in trazos for x in t[0]]
    ys = [y for t in trazos for y in t[1]]
    return min(xs), min(ys), max(xs), max(ys)


def normalizar(trazos, cfg: Config, caja_ref=None):
    """Lleva los trazos a coordenadas del lienzo, centrados y con su proporción.

    Con caja_ref=None la normalización es CAUSAL: usa solo la caja de los trazos
    recibidos (el prefijo). Pasar la caja del dibujo completo solo sirve como
    referencia retrospectiva (por ejemplo, para la figura de prefijos).
    """
    L = cfg.lienzo
    m = cfg.margen * L / cfg.resolucion            # margen en píxeles del lienzo
    x0, y0, x1, y1 = caja_ref if caja_ref is not None else caja(trazos)
    w, h = x1 - x0, y1 - y0
    util = L - 1 - 2 * m
    if max(w, h) == 0:                             # un solo punto o caja de tamaño 0
        a, ox, oy = 0.0, L / 2, L / 2
    else:
        a = util / max(w, h)                       # un único factor de escala
        ox = m + (util - a * w) / 2                # desplazamientos para centrar
        oy = m + (util - a * h) / 2
    return [[[a * (x - x0) + ox for x in t[0]], [a * (y - y0) + oy for y in t[1]]]
            for t in trazos]


def rasterizar(trazos, cfg: Config, caja_ref=None) -> np.ndarray:
    """Imagen uint8 de R x R: fondo 0 y tinta cerca de 255."""
    L = cfg.lienzo
    img = Image.new("L", (L, L), 0)
    dib = ImageDraw.Draw(img)
    r = cfg.grosor / 2
    for xs, ys in normalizar(trazos, cfg, caja_ref):
        puntos = list(zip(xs, ys))
        if len(puntos) > 1:
            dib.line(puntos, fill=255, width=cfg.grosor, joint="curve")
        for x, y in (puntos[0], puntos[-1]):       # extremos redondeados
            dib.ellipse([x - r, y - r, x + r, y + r], fill=255)
    img = img.resize((cfg.resolucion, cfg.resolucion), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


def prefijos_de_dibujo(trazos, cfg: Config) -> np.ndarray:
    """Las K imágenes I_1, ..., I_K de un dibujo (cada una con normalización causal)."""
    return np.stack([rasterizar(trazos[:k], cfg) for k in range(1, len(trazos) + 1)])


# --- Caché de todas las imágenes de una partición ------------------------------

def _trabajo(args):
    trazos, cfg = args
    return prefijos_de_dibujo(trazos, cfg)


def construir_cache(dibujos, indices, cfg: Config, procesos: int = 2):
    """Rasteriza TODOS los prefijos de los dibujos indicados.

    Devuelve un diccionario con:
      imagenes  (N_prefijos, R, R) uint8, los prefijos de cada dibujo seguidos;
      inicio    (N_dibujos,) posición del prefijo k=1 de cada dibujo;
      K         (N_dibujos,) número de trazos de cada dibujo;
      etiqueta  (N_dibujos,) categoría 0..29;
      indice    (N_dibujos,) índice del dibujo en la lista original.
    El prefijo k del dibujo i está en imagenes[inicio[i] + k - 1], y el dibujo
    completo en imagenes[inicio[i] + K[i] - 1].
    """
    tareas = [(dibujos[i]["trazos"], cfg) for i in indices]
    if procesos > 1:
        with Pool(procesos) as p:
            bloques = p.map(_trabajo, tareas, chunksize=64)
    else:
        bloques = [_trabajo(t) for t in tareas]
    K = np.array([len(b) for b in bloques], dtype=np.int32)
    inicio = np.concatenate([[0], np.cumsum(K)[:-1]]).astype(np.int64)
    return {"imagenes": np.concatenate(bloques),
            "inicio": inicio, "K": K,
            "etiqueta": np.array([dibujos[i]["etiqueta"] for i in indices], dtype=np.int64),
            "indice": np.asarray(indices, dtype=np.int64)}


def guardar_cache(cache, ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ruta, **cache)


def cargar_cache(ruta: Path):
    with np.load(ruta) as z:
        return {k: z[k] for k in z.files}


def obtener_cache(dibujos, indices, nombre, cfg: Config, procesos: int = 2):
    """Carga la caché si ya existe en disco (por ejemplo en Drive); si no, la crea.

    El nombre del archivo lleva los parámetros que cambian las imágenes: si el
    grupo cambia el grosor o la resolución, se crea una caché nueva.
    """
    ruta = Path(cfg.ruta_cache) / (f"{nombre}_n{cfg.dibujos_por_categoria}_r{cfg.resolucion}"
                                   f"_l{cfg.lienzo}_g{cfg.grosor}_m{cfg.margen}"
                                   f"_s{cfg.semilla_particion}.npz")
    if ruta.exists():
        return cargar_cache(ruta)
    cache = construir_cache(dibujos, indices, cfg, procesos)
    guardar_cache(cache, ruta)
    return cache


def indices_completos(cache) -> np.ndarray:
    """Posición del dibujo completo (último prefijo) de cada dibujo."""
    return cache["inicio"] + cache["K"] - 1


def indices_fraccion(cache, f: float) -> np.ndarray:
    """Posición del prefijo con k = max(1, ceil(f*K)) trazos de cada dibujo."""
    k = np.array([k_de_fraccion(int(K), f) for K in cache["K"]])
    return cache["inicio"] + k - 1
