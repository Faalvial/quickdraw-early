"""Tablas del paper en LaTeX, escritas directamente desde los resultados.

Así los números no se copian a mano: en la semana 13 basta con volver a
generar las tablas con la partición de prueba. Ver docs/05_metricas.md.

Cada función recibe lo que ya guardó experimentos.py y escribe un .tex con un
`tabular` listo para `\\input{}` dentro de un `table` del paper.
"""
import json
from pathlib import Path

import pandas as pd

ORDEN_MODELOS = ["M1", "M2", "M2-P"]


def _guardar(texto: str, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return texto


def _num(x, dec=1):
    """Número con coma decimal, como en el resto del informe."""
    return f"{x:.{dec}f}".replace(".", ",")


def busqueda(tabla: pd.DataFrame, ruta):
    """Tabla 1: una fila por combinación de la búsqueda, ordenada por exactitud."""
    filas = []
    for _, f in tabla.sort_values(["modelo", "val_exactitud"], ascending=[False, False]).iterrows():
        opt = "Adam" if f["optimizador"] == "adam" else "SGD"
        filas.append(f"{f['modelo']} & {opt} & {_num(f['lr'] * 1000, 1)} & "
                     f"{_num(f['dropout'], 1)} & {_num(100 * f['val_exactitud'])} \\\\")
    texto = ("\\begin{tabular}{llrrr}\n\\toprule\n"
             "Modelo & Optimizador & $\\eta$ ($\\times 10^{-3}$) & Dropout & Exactitud val. (\\%) \\\\\n"
             "\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")
    return _guardar(texto, ruta)


def modelos(metricas: pd.DataFrame, ruta, grupo: str = "todas"):
    """Tabla 2: exactitud final, top-3 y AUC progresiva, media ± desviación entre semillas."""
    m = metricas[metricas["grupo"] == grupo]
    filas = []
    for nombre in ORDEN_MODELOS:
        d = m[m["modelo"] == nombre]
        if d.empty:
            continue
        celdas = []
        for col, escala in (("exactitud_final", 100), ("top3_final", 100), ("auc_progresiva", 1)):
            v = d[col] * escala
            dec = 1 if escala == 100 else 3
            celdas.append(_num(v.mean(), dec) + ("" if len(v) < 2 else f" $\\pm$ {_num(v.std(), dec)}"))
        filas.append(f"{nombre} & {len(d)} & " + " & ".join(celdas) + " \\\\")
    texto = ("\\begin{tabular}{lrrrr}\n\\toprule\n"
             "Modelo & Semillas & Exactitud (\\%) & Top-3 (\\%) & AUC$_{\\mathrm{prog}}$ \\\\\n"
             "\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")
    return _guardar(texto, ruta)


def hipotesis(resultados, ruta):
    """Tabla de hipótesis: el valor que decide cada criterio, por semilla, y el veredicto."""
    claves = {"H1": ("caida_puntos", "caída (pts)", 1), "H2": ("ic_inf", "IC inf. $\\Delta$AUC", 3),
              "H4": ("rho", "$\\rho$", 2), "H5": ("ic_inf", "IC inf. prop.", 3)}
    filas = []
    for r in resultados:
        clave, nombre, dec = claves[r["hipotesis"]]
        valores = "; ".join(_num(f[clave], dec) for f in r["por_semilla"])
        filas.append(f"{r['hipotesis']} & {nombre} & {valores} & {r['veredicto']} \\\\")
    texto = ("\\begin{tabular}{lllr}\n\\toprule\n"
             "Hip. & Criterio & Valor por semilla & Veredicto \\\\\n"
             "\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")
    return _guardar(texto, ruta)


def desde_resultados(ruta_resultados, particion: str = "validacion"):
    """Regenera las tres tablas a partir de lo que ya está en disco."""
    res = Path(ruta_resultados)
    out = res / particion / "tablas"
    busqueda(pd.read_csv(res / "tabla_busqueda.csv"), out / "tabla_busqueda.tex")
    modelos(pd.read_csv(res / particion / "metricas.csv"), out / "tabla_modelos.tex")
    hipotesis(json.loads((res / particion / "hipotesis.json").read_text()), out / "tabla_hipotesis.tex")
    return out
