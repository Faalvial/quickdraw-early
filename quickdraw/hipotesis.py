"""Evaluación de H1, H2, H4 y H5 contra los criterios fijados de antemano.

Cada hipótesis se evalúa en cada semilla y se CONFIRMA solo si el criterio se
cumple en las tres. Ver docs/06_estadistica_e_hipotesis.md.

`preds` es un diccionario {semilla: (pred, top3)} con las predicciones de un
modelo para todos los prefijos de la caché que se evalúa (validación o prueba).
"""
import numpy as np

from . import metricas
from .analisis import confundibilidad, mediana_fraccion_estrella, vectores_cnn
from .estadistica import bootstrap, mcnemar, spearman
from .raster import indices_completos


def _veredicto(filas, clave="cumple"):
    return "confirmada" if all(f[clave] for f in filas) else "refutada"


def h1(cache, preds_m2, cfg):
    """La exactitud a f = 0,5 cae más de 10 puntos respecto a f = 1, y la caída
    es significativa (McNemar, p < 0,05)."""
    m = metricas.mascara_progresiva(cache, cfg.min_trazos_progresivo)
    filas = []
    for s, (pred, _) in sorted(preds_m2.items()):
        c = metricas.correctos_por_fraccion(cache, pred, (0.5, 1.0))[m]
        caida = 100 * (c[:, 1].mean() - c[:, 0].mean())
        prueba = mcnemar(c[:, 1], c[:, 0])
        filas.append({"semilla": s, "acc_f50": c[:, 0].mean(), "acc_f100": c[:, 1].mean(),
                      "caida_puntos": caida, "mcnemar_p": prueba["p"],
                      "cumple": caida > 10 and prueba["p"] < 0.05})
    return {"hipotesis": "H1", "veredicto": _veredicto(filas), "por_semilla": filas}


def h2(cache, preds_m2, preds_m2p, cfg):
    """La AUC progresiva de M2-P supera a la de M2 (IC 95 % bootstrap de la
    diferencia por encima de 0) y la exactitud final no cae más de 1 punto."""
    m = metricas.mascara_progresiva(cache, cfg.min_trazos_progresivo)
    fin = indices_completos(cache)
    filas = []
    for s in sorted(preds_m2):
        p2, p2p = preds_m2[s][0], preds_m2p[s][0]
        auc2 = metricas.auc_por_dibujo(metricas.correctos_por_fraccion(cache, p2)[m])
        auc2p = metricas.auc_por_dibujo(metricas.correctos_por_fraccion(cache, p2p)[m])
        dif, inf, sup = bootstrap(auc2p - auc2, cfg.bootstrap_remuestreos, cfg.semilla_bootstrap)
        final2 = (p2[fin] == cache["etiqueta"]).mean()
        final2p = (p2p[fin] == cache["etiqueta"]).mean()
        caida = 100 * (final2 - final2p)
        filas.append({"semilla": s, "auc_M2": auc2.mean(), "auc_M2P": auc2p.mean(),
                      "dif_auc": dif, "ic_inf": inf, "ic_sup": sup,
                      "final_M2": final2, "final_M2P": final2p, "caida_final_puntos": caida,
                      "cumple": inf > 0 and caida <= 1})
    return {"hipotesis": "H2", "veredicto": _veredicto(filas), "por_semilla": filas}


def h5(cache, preds_m2p, cfg):
    """En los dibujos reconocidos con K >= 3, k* < K en más del 50 % de los
    casos (IC 95 % bootstrap por encima del 50 %). Toda la muestra es de
    dibujos reconocidos por el juego."""
    m = metricas.mascara_progresiva(cache, cfg.min_trazos_progresivo)
    filas = []
    for s, (pred, _) in sorted(preds_m2p.items()):
        ks = metricas.k_estrella(cache, pred)[m]
        antes = (ks > 0) & (ks < cache["K"][m])
        prop, inf, sup = bootstrap(antes, cfg.bootstrap_remuestreos, cfg.semilla_bootstrap)
        filas.append({"semilla": s, "proporcion_k_menor_K": prop, "ic_inf": inf, "ic_sup": sup,
                      "sin_k_estrella": float((ks == 0).mean()), "cumple": inf > 0.5})
    return {"hipotesis": "H5", "veredicto": _veredicto(filas), "por_semilla": filas}


def h4(cache_val, modelos_m2, cache_eval, preds_m2p, cfg):
    """Spearman rho > 0 con p < 0,05 entre la confundibilidad de cada categoría
    (en el espacio GAP de M2, con dibujos completos de validación) y la mediana
    de k*/K de M2-P en esa categoría."""
    filas = []
    for s in sorted(preds_m2p):
        v = vectores_cnn(modelos_m2[s], cache_val, 1.0)
        conf = confundibilidad(v, cache_val["etiqueta"], cfg.vecinos_knn)
        ks = metricas.k_estrella(cache_eval, preds_m2p[s][0])
        med = mediana_fraccion_estrella(cache_eval, ks)
        r = spearman(conf, med)
        filas.append({"semilla": s, "rho": r["rho"], "p": r["p"],
                      "cumple": r["rho"] > 0 and r["p"] < 0.05,
                      "confundibilidad": conf.tolist(), "mediana_k_estrella_K": med.tolist()})
    return {"hipotesis": "H4", "veredicto": _veredicto(filas), "por_semilla": filas}
