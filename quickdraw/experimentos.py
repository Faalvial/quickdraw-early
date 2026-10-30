"""Flujo completo del proyecto, en etapas. El notebook llama a estas funciones
en orden. Cada etapa guarda su resultado en disco y, si ya existe, lo reutiliza:
así se puede retomar el trabajo cuando Colab corta la sesión.

Ver docs/00_como_ejecutar.md.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import analisis, datos, figuras, hipotesis, metricas, tablas
from .config import FRACCIONES, PARES, Config
from .entrenar import (DISPOSITIVO, busqueda, entrenar_o_cargar, mejor_combinacion,
                       predecir, rejilla_m1, rejilla_m2, ruta_modelo)
from .raster import indices_completos, obtener_cache


# --- 1. Datos -------------------------------------------------------------------

def preparar_datos(cfg: Config, procesos: int = 2):
    """Lee la muestra, hace las particiones y rasteriza todos los prefijos.

    La caché de prueba solo se construye si cfg.usar_prueba es True.
    """
    res = Path(cfg.ruta_resultados)
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dibujos = datos.cargar_muestra(cfg)
    partes = datos.particionar(dibujos, cfg)
    datos.guardar_particiones(dibujos, partes, res / "particiones")
    pd.DataFrame(datos.tabla_hechos(dibujos)).to_csv(res / "tabla_hechos.csv", index=False)
    print(f"{len(dibujos)} dibujos leídos; particiones: "
          + ", ".join(f"{k} {len(v)}" for k, v in partes.items()))
    caches = {}
    nombres = ["entrenamiento", "validacion"] + (["prueba"] if cfg.usar_prueba else [])
    for nombre in nombres:
        caches[nombre] = obtener_cache(dibujos, partes[nombre], nombre, cfg, procesos)
        print(f"  caché {nombre}: {len(caches[nombre]['imagenes'])} prefijos")
    print(f"Datos listos en {time.time() - t0:.0f} s")
    return {"dibujos": dibujos, "partes": partes, "caches": caches}


# --- 2. Búsqueda de hiperparámetros -------------------------------------------------

def buscar_hiperparametros(cfg: Config, D):
    ent, val = D["caches"]["entrenamiento"], D["caches"]["validacion"]
    filas_m2 = busqueda("M2", ent, val, cfg, rejilla_m2(cfg))
    filas_m1 = busqueda("M1", ent, val, cfg, rejilla_m1(cfg))
    tabla = pd.DataFrame([{"modelo": m, **f["hp"], "val_exactitud": f["val_exactitud"]}
                          for m, filas in (("M2", filas_m2), ("M1", filas_m1)) for f in filas])
    tabla.sort_values(["modelo", "val_exactitud"], ascending=[True, False]).to_csv(
        Path(cfg.ruta_resultados) / "tabla_busqueda.csv", index=False)
    hp = {"M1": mejor_combinacion(filas_m1), "M2": mejor_combinacion(filas_m2)}
    hp["M2-P"] = hp["M2"]          # M2-P hereda la configuración de M2 (sección 3 del plan)
    (Path(cfg.ruta_resultados) / "hiperparametros_elegidos.json").write_text(json.dumps(hp, indent=1))
    return hp, tabla


# --- 3. Entrenamiento -------------------------------------------------------------

def entrenar_modelos(cfg: Config, D, hp, nombres=("M1", "M2", "M2-P")):
    ent, val = D["caches"]["entrenamiento"], D["caches"]["validacion"]
    modelos = {}
    for nombre in nombres:
        semillas = cfg.semillas_m1 if nombre == "M1" else cfg.semillas
        for s in semillas:
            modelos[(nombre, s)], _ = entrenar_o_cargar(nombre, ent, val, cfg, hp[nombre], s)
    return modelos


# --- 4. Predicciones ----------------------------------------------------------------

def predecir_todo(cfg: Config, D, modelos, particion: str):
    """Predicciones de cada modelo para TODOS los prefijos de la partición.

    Se guardan en disco; si el modelo se reentrenó después, se recalculan.
    """
    if particion == "prueba" and not cfg.usar_prueba:
        raise RuntimeError("El conjunto de prueba está bloqueado (cfg.usar_prueba=False).")
    cache = D["caches"][particion]
    carpeta = Path(cfg.ruta_resultados) / f"predicciones_{particion}"
    carpeta.mkdir(parents=True, exist_ok=True)
    preds = {}
    for (nombre, s), modelo in modelos.items():
        ruta = carpeta / f"{nombre}_s{s}.npz"
        pesos = ruta_modelo(cfg, nombre, s) / "modelo.pt"
        if ruta.exists() and (not pesos.exists() or ruta.stat().st_mtime >= pesos.stat().st_mtime):
            z = np.load(ruta)
            preds[(nombre, s)] = (z["pred"], z["top3"])
            continue
        pred, top3 = predecir(modelo, cache["imagenes"])
        np.savez_compressed(ruta, pred=pred, top3=top3)
        preds[(nombre, s)] = (pred, top3)
    return preds


def _por_semilla(preds, nombre):
    return {s: p for (n, s), p in preds.items() if n == nombre}


# --- 5. Evaluación, hipótesis y figuras ---------------------------------------------

def evaluar(cfg: Config, D, modelos, preds, particion: str):
    """Tablas, hipótesis y figuras de una partición ('validacion' o 'prueba')."""
    cache = D["caches"][particion]
    val = D["caches"]["validacion"]
    out = Path(cfg.ruta_resultados) / particion
    out.mkdir(parents=True, exist_ok=True)

    # Tabla de métricas por modelo, semilla y grupo de categorías
    filas = []
    for (nombre, s), (pred, top3) in sorted(preds.items()):
        for f in metricas.resumen(cache, pred, top3, cfg.min_trazos_progresivo):
            filas.append({"modelo": nombre, "semilla": s, **f})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(out / "metricas.csv", index=False)
    cols = ["exactitud_final", "top3_final", "auc_progresiva"]
    agregada = tabla.groupby(["modelo", "grupo"])[cols].agg(["mean", "std"]).round(4)
    agregada.to_csv(out / "metricas_media_desviacion.csv")

    # Hipótesis
    m2, m2p = _por_semilla(preds, "M2"), _por_semilla(preds, "M2-P")
    modelos_m2 = {s: modelos[("M2", s)] for s in m2}
    res_h = [hipotesis.h1(cache, m2, cfg), hipotesis.h2(cache, m2, m2p, cfg),
             hipotesis.h4(val, modelos_m2, cache, m2p, cfg), hipotesis.h5(cache, m2p, cfg)]
    (out / "hipotesis.json").write_text(json.dumps(res_h, indent=1, default=float))
    veredictos = pd.DataFrame([{"hipotesis": r["hipotesis"], "veredicto": r["veredicto"]}
                               for r in res_h])

    # Figuras
    prog = metricas.mascara_progresiva(cache, cfg.min_trazos_progresivo)
    curvas = {}
    for nombre in ("M1", "M2", "M2-P"):
        ps = _por_semilla(preds, nombre)
        if ps:
            curvas[nombre] = np.stack([metricas.correctos_por_fraccion(cache, p)[prog].mean(0)
                                       for p, _ in ps.values()])
    figuras.curvas(curvas, out / "fig_curvas_fraccion.png")
    figuras.por_k({n: metricas.exactitud_por_k(cache, _por_semilla(preds, n)[min(_por_semilla(preds, n))][0])
                   for n in curvas}, out / "fig_exactitud_por_k.png")
    s0 = min(m2p)
    pred0 = m2p[s0][0]
    for f in (0.3, 1.0):
        figuras.confusion(metricas.matriz_confusion(cache, pred0, f),
                          f"M2-P con el {int(f * 100)} % del dibujo",
                          out / f"fig_confusion_f{int(f * 100)}.png")
    h4 = res_h[2]["por_semilla"][0]
    figuras.h4(np.array(h4["confundibilidad"]), np.array(h4["mediana_k_estrella_K"]),
               h4["rho"], h4["p"], out / "fig_h4.png")
    ks = metricas.k_estrella(cache, pred0)
    figuras.h5(cache["K"][prog], ks[prog], out / "fig_h5.png")
    mapas = {}
    for f in (0.3, 1.0):
        v = analisis.vectores_cnn(modelos[("M2-P", s0)], cache, f)
        mapas[f"M2-P, {int(f * 100)} % del dibujo"] = analisis.mapa_tsne(
            v, cache["etiqueta"], cfg.tsne_por_categoria, cfg.tsne_perplejidad)
    figuras.tsne(mapas, out / "fig_tsne.png")
    pd.DataFrame(analisis.separacion_de_pares(cache, pred0, PARES, FRACCIONES)).to_csv(
        out / "separacion_pares.csv", index=False)
    ejemplos = _ejemplos(cache)
    figuras.trazo_a_trazo(cache, {n: _por_semilla(preds, n)[min(_por_semilla(preds, n))][0]
                                  for n in curvas}, ejemplos, out / "fig_trazo_a_trazo.png")
    tablas.desde_resultados(cfg.ruta_resultados, particion)   # tablas LaTeX del paper
    print(f"Resultados de {particion} guardados en {out}")
    return {"tabla": tabla, "agregada": agregada, "hipotesis": res_h, "veredictos": veredictos}


def _ejemplos(cache, n: int = 4, semilla: int = 0):
    """Un dibujo de 5 a 10 trazos de algunas categorías de los pares."""
    rng = np.random.default_rng(semilla)
    elegidos = []
    for etiqueta in (0, 1, 2, 3):
        cand = np.nonzero((cache["etiqueta"] == etiqueta) & (cache["K"] >= 5) & (cache["K"] <= 10))[0]
        if len(cand):
            elegidos.append(int(rng.choice(cand)))
    return elegidos[:n]


# --- 6. Latencia --------------------------------------------------------------------

def latencia(modelos, repeticiones: int = 200):
    """Milisegundos por predicción en CPU con un solo dibujo (meta: < 50 ms)."""
    filas = []
    x = torch.zeros(1, 1, 28, 28)
    for nombre in ("M1", "M2", "M2-P"):
        claves = [k for k in modelos if k[0] == nombre]
        if not claves:
            continue
        m = modelos[claves[0]].to("cpu").eval()
        with torch.no_grad():
            for _ in range(10):
                m(x)
            t0 = time.perf_counter()
            for _ in range(repeticiones):
                m(x)
        filas.append({"modelo": nombre,
                      "ms_por_prediccion": round(1000 * (time.perf_counter() - t0) / repeticiones, 3)})
        m.to(DISPOSITIVO)
    return pd.DataFrame(filas)
