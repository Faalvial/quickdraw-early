"""Lectura de la muestra, particiones y tabla de hechos.

Ver docs/01_datos.md.
"""
import gzip
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from .config import CATEGORIAS, CATEGORIAS_PARES, Config


def cargar_muestra(cfg: Config, categorias=CATEGORIAS):
    """Lee los dibujos reconocidos de cada categoría.

    Devuelve una lista de diccionarios con: key_id, categoria, etiqueta (0..29),
    pais y trazos. Cada trazo es un par de listas [xs, ys] (formato simplified).
    """
    dibujos = []
    for etiqueta, cat in enumerate(categorias):
        ruta = Path(cfg.ruta_muestra) / f"{cat}.ndjson.gz"
        if not ruta.exists():
            raise FileNotFoundError(
                f"Falta {ruta}. Suban la carpeta muestra_30 a Drive o ejecuten "
                "datos/download_sample.py (ver docs/01_datos.md).")
        with gzip.open(ruta, "rt") as f:
            for i, linea in enumerate(f):
                if i >= cfg.dibujos_por_categoria:
                    break
                d = json.loads(linea)
                dibujos.append({"key_id": d["key_id"], "categoria": cat,
                                "etiqueta": etiqueta, "pais": d["countrycode"],
                                "trazos": d["drawing"]})
    return dibujos


def particionar(dibujos, cfg: Config):
    """Divide 70/15/15 por dibujo, con la misma proporción en cada categoría.

    Se divide ANTES de generar prefijos: todos los prefijos de un dibujo quedan
    en la misma partición. Devuelve {"entrenamiento": [...], "validacion": [...],
    "prueba": [...]} con índices sobre `dibujos`.
    """
    rng = random.Random(cfg.semilla_particion)
    por_categoria = {}
    for i, d in enumerate(dibujos):
        por_categoria.setdefault(d["etiqueta"], []).append(i)
    partes = {"entrenamiento": [], "validacion": [], "prueba": []}
    p_ent, p_val, _ = cfg.proporciones
    for etiqueta in sorted(por_categoria):
        idx = por_categoria[etiqueta][:]
        rng.shuffle(idx)
        n = len(idx)
        n_ent, n_val = round(n * p_ent), round(n * p_val)
        partes["entrenamiento"] += idx[:n_ent]
        partes["validacion"] += idx[n_ent:n_ent + n_val]
        partes["prueba"] += idx[n_ent + n_val:]
    return {k: np.array(sorted(v)) for k, v in partes.items()}


def guardar_particiones(dibujos, partes, ruta: Path):
    """Guarda las listas de key_id de cada partición, para la reproducibilidad."""
    ruta.mkdir(parents=True, exist_ok=True)
    for nombre, idx in partes.items():
        with open(ruta / f"key_ids_{nombre}.json", "w") as f:
            json.dump([dibujos[i]["key_id"] for i in idx], f)


def tabla_hechos(dibujos, categorias=CATEGORIAS):
    """Número de dibujos, mediana de trazos y país más frecuente por categoría."""
    filas = []
    for cat in categorias:
        ds = [d for d in dibujos if d["categoria"] == cat]
        trazos = [len(d["trazos"]) for d in ds]
        pais, n = Counter(d["pais"] for d in ds).most_common(1)[0]
        filas.append({"categoria": cat,
                      "grupo": "par" if cat in CATEGORIAS_PARES else "azar",
                      "dibujos": len(ds),
                      "mediana_trazos": statistics.median(trazos),
                      "menos_de_3_trazos": sum(t < 3 for t in trazos),
                      "pais_mas_frecuente": pais,
                      "porcentaje_pais": round(100 * n / len(ds), 1)})
    return filas
