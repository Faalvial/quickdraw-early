"""Entrenamiento, predicción y búsqueda de hiperparámetros.
Ver docs/04_entrenamiento_y_busqueda.md.

Los datos viven en memoria como tensores uint8 (las cachés de raster.py), así
que no hace falta DataLoader: cada lote se arma indexando el tensor.
"""
import json
import random
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import metricas
from .config import Config
from .modelos import crear_modelo, preparar_entrada
from .raster import indices_completos

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(semilla: int):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def indices_de_epoca(cache, esquema: str, rng: np.random.Generator,
                     prob_completo: float = 0.5) -> np.ndarray:
    """Qué imagen de cada dibujo se usa en esta época.

    'completos': siempre el dibujo completo (M1 y M2).
    'prefijos' : un prefijo al azar con k uniforme entre 1 y K (M2-P). Cada
                 dibujo aporta una imagen por época, sin dar más peso a los
                 dibujos con más trazos.
    'mixto'    : como 'prefijos', pero con probabilidad prob_completo se usa
                 el dibujo completo (alternativa si M2-P pierde exactitud final).
    """
    if esquema == "completos":
        return indices_completos(cache)
    if esquema == "prefijos":
        k = rng.integers(1, cache["K"] + 1)          # entre 1 y K, ambos incluidos
        return cache["inicio"] + k - 1
    if esquema == "mixto":
        k = rng.integers(1, cache["K"] + 1)
        k = np.where(rng.random(len(k)) < prob_completo, cache["K"], k)
        return cache["inicio"] + k - 1
    raise ValueError(f"Esquema desconocido: {esquema}")


@torch.no_grad()
def predecir(modelo, imagenes, lote: int = 1024, vectores: bool = False):
    """Predicción para un arreglo de imágenes uint8 (N, 28, 28).

    Devuelve (pred, top3) o, con vectores=True, los vectores GAP de la CNN.
    """
    modelo.eval()
    salida_pred, salida_top3, salida_vec = [], [], []
    tensor = torch.from_numpy(np.ascontiguousarray(imagenes))
    for i in range(0, len(tensor), lote):
        x = preparar_entrada(tensor[i:i + lote].to(DISPOSITIVO))
        if vectores:
            salida_vec.append(modelo.vectores(x).float().cpu())
            continue
        logits = modelo(x).float()
        top3 = logits.topk(3, dim=1).indices
        salida_pred.append(top3[:, 0].cpu())
        salida_top3.append(top3.cpu())
    if vectores:
        return torch.cat(salida_vec).numpy()
    return (torch.cat(salida_pred).numpy().astype(np.int16),
            torch.cat(salida_top3).numpy().astype(np.int16))


def _submuestra(cache, n: int, semilla: int = 0):
    """Una caché con solo n dibujos (para medir la AUC en validación cada época)."""
    if n >= len(cache["K"]):
        return cache
    rng = np.random.default_rng(semilla)
    elegidos = np.sort(rng.choice(len(cache["K"]), size=n, replace=False))
    bloques = [cache["imagenes"][cache["inicio"][i]:cache["inicio"][i] + cache["K"][i]]
               for i in elegidos]
    K = cache["K"][elegidos]
    return {"imagenes": np.concatenate(bloques),
            "inicio": np.concatenate([[0], np.cumsum(K)[:-1]]).astype(np.int64),
            "K": K, "etiqueta": cache["etiqueta"][elegidos],
            "indice": cache["indice"][elegidos]}


def objetivo_validacion(modelo, cache_val, objetivo: str, min_trazos: int = 3) -> float:
    """'final': exactitud con el dibujo completo. 'auc': AUC progresiva."""
    if objetivo == "final":
        pred, _ = predecir(modelo, cache_val["imagenes"][indices_completos(cache_val)])
        return float((pred == cache_val["etiqueta"]).mean())
    pred, _ = predecir(modelo, cache_val["imagenes"])
    m = metricas.mascara_progresiva(cache_val, min_trazos)
    correctos = metricas.correctos_por_fraccion(cache_val, pred)[m]
    return metricas.auc_progresiva(correctos.mean(0))


def entrenar(nombre_modelo: str, cache_ent, cache_val, cfg: Config, *, esquema: str,
             optimizador: str = "adam", lr: float = 1e-3, dropout: float = 0.0,
             semilla: int = 0, epocas: int | None = None, fraccion_datos: float = 1.0,
             objetivo: str = "final", ruta: Path | None = None, verbose: bool = True):
    """Entrena un modelo y devuelve (modelo con los mejores pesos, historial).

    - Entropía cruzada, lotes de cfg.lote, calendario coseno.
    - Parada temprana: se detiene si el objetivo de validación no mejora en
      cfg.paciencia épocas, y se quedan los pesos de la mejor época.
    - Si se da `ruta`, guarda un checkpoint por época y retoma desde él si la
      sesión de Colab se cortó.
    """
    epocas = epocas or cfg.epocas_max
    fijar_semilla(semilla)
    modelo = crear_modelo(nombre_modelo, dropout).to(DISPOSITIVO)
    if optimizador == "adam":
        opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    elif optimizador == "sgd":
        opt = torch.optim.SGD(modelo.parameters(), lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(optimizador)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    usar_amp = DISPOSITIVO.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)
    perdida = nn.CrossEntropyLoss()

    rng = np.random.default_rng(semilla)
    n = len(cache_ent["K"])
    subconjunto = (np.sort(rng.choice(n, size=int(n * fraccion_datos), replace=False))
                   if fraccion_datos < 1 else np.arange(n))
    imagenes = torch.from_numpy(cache_ent["imagenes"])
    etiquetas = torch.from_numpy(cache_ent["etiqueta"])
    val = _submuestra(cache_val, cfg.dibujos_val_auc) if objetivo == "auc" else cache_val

    historial, mejor, mejor_estado, sin_mejora, inicio_epoca = [], -1.0, None, 0, 0
    ckpt = Path(ruta) / "checkpoint.pt" if ruta else None
    if ckpt and ckpt.exists():                         # retomar una sesión cortada
        estado = torch.load(ckpt, map_location=DISPOSITIVO, weights_only=False)
        modelo.load_state_dict(estado["modelo"]); opt.load_state_dict(estado["opt"])
        sched.load_state_dict(estado["sched"]); historial = estado["historial"]
        mejor, mejor_estado, sin_mejora = estado["mejor"], estado["mejor_estado"], estado["sin_mejora"]
        inicio_epoca = len(historial)
        rng = estado["rng"]

    for epoca in range(inicio_epoca, epocas):
        if sin_mejora >= cfg.paciencia:
            break
        t0 = time.time()
        modelo.train()
        idx_img = indices_de_epoca(cache_ent, esquema, rng, cfg.m2p_prob_completo)[subconjunto]
        etiq = etiquetas[subconjunto]
        orden = torch.from_numpy(rng.permutation(len(idx_img)))
        idx_img = torch.from_numpy(idx_img)[orden]
        etiq = etiq[orden]
        total = 0.0
        for i in range(0, len(idx_img), cfg.lote):
            x = preparar_entrada(imagenes[idx_img[i:i + cfg.lote]].to(DISPOSITIVO, non_blocking=True))
            y = etiq[i:i + cfg.lote].to(DISPOSITIVO, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DISPOSITIVO.type, enabled=usar_amp):
                loss = perdida(modelo(x), y)
            escalador.scale(loss).backward()
            escalador.step(opt)
            escalador.update()
            total += loss.item() * len(y)
        sched.step()
        valor = objetivo_validacion(modelo, val, objetivo, cfg.min_trazos_progresivo)
        historial.append({"epoca": epoca + 1, "perdida": total / len(idx_img),
                          "validacion": valor, "segundos": round(time.time() - t0, 1)})
        if valor > mejor:
            mejor, sin_mejora = valor, 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
        if verbose:
            h = historial[-1]
            print(f"  época {h['epoca']:2d}  pérdida {h['perdida']:.3f}  "
                  f"val {objetivo} {h['validacion']:.4f}  ({h['segundos']} s)")
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "historial": historial,
                        "mejor": mejor, "mejor_estado": mejor_estado,
                        "sin_mejora": sin_mejora, "rng": rng}, ckpt)

    modelo.load_state_dict(mejor_estado)
    return modelo, historial


def ruta_modelo(cfg: Config, nombre: str, semilla: int) -> Path:
    return Path(cfg.ruta_modelos) / f"{nombre}_s{semilla}"


def huella_datos(cfg: Config) -> dict:
    """Lo que define las imágenes con que se entrena. Si cambia, los modelos y
    las búsquedas guardados ya no corresponden a los datos."""
    return {"dibujos_por_categoria": cfg.dibujos_por_categoria, "resolucion": cfg.resolucion,
            "lienzo": cfg.lienzo, "grosor": cfg.grosor, "margen": cfg.margen,
            "semilla_particion": cfg.semilla_particion, "proporciones": list(cfg.proporciones)}


def huella_modelo(cfg: Config, hp: dict) -> dict:
    """Hiperparámetros + datos + reglas de entrenamiento de un modelo guardado."""
    return {**hp, **huella_datos(cfg), "lote": cfg.lote, "epocas_max": cfg.epocas_max,
            "paciencia": cfg.paciencia, "dibujos_val_auc": cfg.dibujos_val_auc}


def entrenar_o_cargar(nombre: str, cache_ent, cache_val, cfg: Config, hp: dict, semilla: int):
    """Entrena el modelo si no existe su archivo final; si existe, solo lo carga.

    hp: {"optimizador", "lr", "dropout"}. M2-P usa el esquema 'prefijos' y la
    AUC progresiva como objetivo; M1 y M2 usan dibujos completos.
    """
    carpeta = ruta_modelo(cfg, nombre, semilla)
    final = carpeta / "modelo.pt"
    esquema, objetivo = (cfg.esquema_m2p, "auc") if nombre == "M2-P" else ("completos", "final")
    huella = huella_modelo(cfg, hp)
    if nombre == "M2-P" and esquema != "prefijos":
        huella = {**huella, "esquema": esquema, "prob_completo": cfg.m2p_prob_completo}
    ruta_hp = carpeta / "hiperparametros.json"
    if ruta_hp.exists() and json.loads(ruta_hp.read_text()) != huella:
        raise RuntimeError(
            f"{carpeta} se entrenó (o empezó a entrenarse) con otra configuración.\n"
            f"  guardada: {json.loads(ruta_hp.read_text())}\n  actual:   {huella}\n"
            "Si el cambio es a propósito, borren esa carpeta (y predicciones_*) para "
            "reentrenar. Ver docs/00_como_ejecutar.md.")
    if final.exists():
        modelo = crear_modelo(nombre, hp["dropout"]).to(DISPOSITIVO)
        modelo.load_state_dict(torch.load(final, map_location=DISPOSITIVO))
        historial = json.loads((carpeta / "historial.json").read_text())
        return modelo, historial
    print(f"Entrenando {nombre}, semilla {semilla}, {hp}")
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_hp.write_text(json.dumps(huella, indent=1))
    modelo, historial = entrenar(nombre, cache_ent, cache_val, cfg, esquema=esquema,
                                 objetivo=objetivo, semilla=semilla, ruta=carpeta, **hp)
    torch.save(modelo.state_dict(), final)
    (carpeta / "historial.json").write_text(json.dumps(historial, indent=1))
    return modelo, historial


def busqueda(nombre: str, cache_ent, cache_val, cfg: Config, rejilla: list[dict]):
    """Prueba cada combinación con pocas épocas y parte de los datos.

    Devuelve una fila por combinación de la rejilla con la mejor exactitud de
    validación. Las filas ya calculadas se leen de disco, así que se puede
    interrumpir; las que se hicieron con otros datos u otro presupuesto de
    búsqueda se ignoran y se vuelven a calcular.
    """
    ruta = Path(cfg.ruta_resultados) / f"busqueda_{nombre}.json"
    contexto = {**huella_datos(cfg), "epocas": cfg.busqueda_epocas,
                "fraccion_datos": cfg.busqueda_fraccion_datos}
    filas = json.loads(ruta.read_text()) if ruta.exists() else []
    filas = [f for f in filas if f.get("contexto") == contexto]
    clave = lambda hp: json.dumps(hp, sort_keys=True)
    hechas = {clave(f["hp"]) for f in filas}
    for hp in rejilla:
        if clave(hp) in hechas:
            continue
        print(f"Búsqueda {nombre}: {hp}")
        _, hist = entrenar(nombre, cache_ent, cache_val, cfg, esquema="completos",
                           objetivo="final", semilla=0, epocas=cfg.busqueda_epocas,
                           fraccion_datos=cfg.busqueda_fraccion_datos, verbose=False, **hp)
        filas.append({"hp": hp, "val_exactitud": max(h["validacion"] for h in hist),
                      "epocas": len(hist), "contexto": contexto})
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(filas, indent=1))
    en_rejilla = {clave(hp) for hp in rejilla}
    return [f for f in filas if clave(f["hp"]) in en_rejilla]


def rejilla_m2(cfg: Config) -> list[dict]:
    return [{"optimizador": o, "lr": lr, "dropout": d}
            for o, lr, d in product(cfg.busqueda_optimizadores, cfg.busqueda_lr,
                                    cfg.busqueda_dropout)]


def rejilla_m1(cfg: Config) -> list[dict]:
    return [{"optimizador": "adam", "lr": lr, "dropout": cfg.m1_dropout}
            for lr in cfg.busqueda_lr]


def mejor_combinacion(filas) -> dict:
    return max(filas, key=lambda f: f["val_exactitud"])["hp"]
