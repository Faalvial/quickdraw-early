"""Duración de los dibujos reconocidos frente a los no reconocidos (formato raw).

Sostiene la premisa de H5: la ronda del juego termina cuando la IA acierta, así
que los dibujos reconocidos deberían durar bastante menos que los 20 s del
límite. Se mide solo en "cat". Ver docs/01_datos.md.
"""
import json
import random
import statistics

import requests

URL_RAW = "https://storage.googleapis.com/quickdraw_dataset/full/raw/{}.ndjson"


def medir_duracion(categoria: str = "cat", n: int = 3000, semilla: int = 2026,
                   max_lineas: int | None = None):
    """Muestreo por reservorio sobre el archivo raw y mediana de la duración.

    En el formato raw cada trazo es [xs, ys, ts], con ts en milisegundos desde
    el primer punto. La duración es el último tiempo del último trazo.
    max_lineas limita la lectura para pruebas rápidas (sesga hacia los dibujos
    más antiguos, porque el archivo está ordenado por fecha).
    """
    rng = random.Random(semilla)
    reservas = {True: [], False: []}
    vistos = {True: 0, False: 0}
    with requests.get(URL_RAW.format(categoria), stream=True, timeout=60) as r:
        r.raise_for_status()
        for i, linea in enumerate(r.iter_lines()):
            if max_lineas and i >= max_lineas:
                break
            d = json.loads(linea)
            g = bool(d["recognized"])
            dur = d["drawing"][-1][2][-1] / 1000
            vistos[g] += 1
            if len(reservas[g]) < n:
                reservas[g].append(dur)
            else:
                j = rng.randrange(vistos[g])
                if j < n:
                    reservas[g][j] = dur

    def resumen(v):
        v = sorted(v)
        return {"dibujos": len(v), "mediana_s": round(statistics.median(v), 2),
                "p90_s": round(v[int(0.9 * (len(v) - 1))], 2)}

    return {"categoria": categoria, "reconocidos": resumen(reservas[True]),
            "no_reconocidos": resumen(reservas[False]),
            "lineas_leidas": vistos[True] + vistos[False]}
