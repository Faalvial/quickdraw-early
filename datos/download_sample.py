import gzip, json, os, random, statistics, urllib.parse
from collections import Counter
import requests

CATEGORIAS = ["cat", "dog", "lion", "tiger", "horse", "zebra", "violin", "cello",
              "bear", "panda", "The Mona Lisa", "alarm clock", "bird", "brain",
              "bridge", "bus", "calculator", "fireplace", "frog", "jail", "map",
              "remote control", "scorpion", "sink", "stop sign", "sun", "teddy-bear",
              "lobster", "whale", "yoga"]   # trumpet salió por mediana 4; lobster fue la siguiente con la semilla 2026
URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"

def muestrear(categoria, n_rec=10000, n_norec=1000, semilla=2026):
    """Recorre el archivo completo y toma una muestra uniforme (Algoritmo R)."""
    rng = random.Random(semilla)
    reservas = {True: [], False: []}          # reconocidos y no reconocidos
    tope = {True: n_rec, False: n_norec}
    vistos = {True: 0, False: 0}
    url = URL.format(urllib.parse.quote(categoria))
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        for linea in r.iter_lines():
            d = json.loads(linea)
            g = d["recognized"]
            vistos[g] += 1
            if len(reservas[g]) < tope[g]:
                reservas[g].append(d)
            else:
                j = rng.randrange(vistos[g])   # entero al azar entre 0 y vistos - 1
                if j < tope[g]:
                    reservas[g][j] = d
    return reservas[True], reservas[False], vistos

def guardar(dibujos, ruta):
    with gzip.open(ruta, "wt") as f:
        for d in dibujos:
            f.write(json.dumps(d) + "\n")

def hechos(categoria, rec, vistos):
    trazos = [len(d["drawing"]) for d in rec]
    paises = Counter(d["countrycode"] for d in rec)
    pais, cuantos = paises.most_common(1)[0]
    return {"categoria": categoria,
            "dibujos_en_archivo": vistos[True] + vistos[False],
            "prop_recognized": round(vistos[True] / (vistos[True] + vistos[False]), 4),
            "mediana_trazos": statistics.median(trazos),
            "prop_menos_de_3_trazos": round(sum(t < 3 for t in trazos) / len(trazos), 4),
            "pais_mas_comun": pais,
            "prop_pais_mas_comun": round(cuantos / len(rec), 4)}

if __name__ == "__main__":
    import csv, sys
    carpeta = sys.argv[1]                       # p. ej. /content/drive/MyDrive/quickdraw/datos/muestra
    os.makedirs(carpeta, exist_ok=True)
    for c in CATEGORIAS:
        ruta = os.path.join(carpeta, c + ".ndjson.gz")
        if os.path.exists(ruta):                # ya estaba: no se descarga de nuevo
            continue
        rec, norec, vistos = muestrear(c)
        guardar(norec, os.path.join(carpeta, c + "_no_reconocidos.ndjson.gz"))
        with open(os.path.join(carpeta, c + "_hechos.json"), "w") as f:
            json.dump(hechos(c, rec, vistos), f)
        guardar(rec, ruta)                      # se guarda al final: si existe, la categoría está completa
        print("listo:", c)
    filas = [json.load(open(os.path.join(carpeta, c + "_hechos.json"))) for c in CATEGORIAS]
    with open(os.path.join(carpeta, "hechos_avance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=filas[0].keys())
        w.writeheader()
        w.writerows(filas)
