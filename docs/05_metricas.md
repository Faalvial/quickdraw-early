# 05 · Métricas progresivas

**Archivos:** `quickdraw/metricas.py`, `quickdraw/figuras.py`
**Responsable en el plan:** Omar
**Teoría:** sección 3.7 del avance (exactitud por fracción, AUC progresiva, k*, matrices de confusión)

Todas las funciones reciben la caché de una partición y la predicción del modelo para **todos** sus prefijos (`pred`, y `top3` con las tres clases más probables). Así un modelo se evalúa una sola vez y las métricas se calculan sobre esas predicciones.

## Qué mide cada función

| Función | Qué devuelve | Dónde aparece |
|---|---|---|
| `correctos_por_fraccion` | Matriz dibujos × 10 fracciones: ¿acertó con el prefijo de k = ⌈f·K⌉ trazos? | Curvas, H1, H2 |
| `top3_por_fraccion` | Lo mismo con top-3 | Tabla de métricas |
| `auc_progresiva` | (1/0,9) · ∫ de 0,1 a 1 de acc(f) df, con trapecios. Vale 1 si acierta siempre | Tabla, H2 |
| `auc_por_dibujo` | La AUC de cada dibujo; su promedio es la AUC del conjunto | Bootstrap de H2 |
| `exactitud_por_k` | Exactitud con exactamente k trazos (k = 1 a 10), entre los dibujos con K ≥ k | `fig_exactitud_por_k.png` |
| `k_estrella` | k*: primer trazo desde el cual la predicción es correcta **y ya no cambia**. 0 si falla con el dibujo completo | H4, H5 |
| `fraccion_estrella` | k*/K; los dibujos sin k* cuentan como 1 (el peor valor) | H4 |
| `matriz_confusion` | Conteos verdadera × predicha en una fracción | `fig_confusion_f30.png`, `fig_confusion_f100.png` |
| `resumen` | Exactitud final, top-3 final, AUC y exactitud por fracción, para todas, las 20 al azar y los 10 de los pares | `metricas.csv` |

Ejemplo de k* (el del avance): con etiqueta "gato" y predicciones perro, gato, perro, gato, gato, k* = 4, porque el modelo cambia de opinión en el trazo 3 y solo desde el 4 se queda en "gato". Lo comprueba `tests/test_metricas.py`.

## Decisiones y por qué

- **Fracciones en vez de número de trazos.** Un 30 % de un dibujo de 3 trazos y de uno de 20 son comparables; "3 trazos" no lo es. Por eso la curva principal es por fracción y la de k fijo es complementaria.
- **Curvas y H5 solo con K ≥ 3.** Con 1 o 2 trazos todas las fracciones caen en el mismo prefijo o casi, y la curva sería plana por construcción (`min_trazos_progresivo`).
- **k* exige estabilidad.** Acertar en el trazo 2 y volver a fallar en el 3 no cuenta como reconocido: en el juego, la IA solo gana si sigue acertando.
- **Pares frente a azar.** Los 10 de los pares deberían tardar más; mostrarlos por separado conecta las métricas con la segunda mitad de la pregunta.

## Figuras (`figuras.py`)

| Figura | Qué muestra |
|---|---|
| `fig_prefijos.png` | Prefijos con normalización causal frente a la caja final |
| `fig_curvas_fraccion.png` | Exactitud por fracción de M1, M2 y M2-P, media ± desviación entre semillas |
| `fig_exactitud_por_k.png` | Exactitud por número de trazos |
| `fig_confusion_f30.png`, `fig_confusion_f100.png` | Matrices de confusión normalizadas por fila, de M2-P (semilla 0) |
| `fig_trazo_a_trazo.png` | Cuatro dibujos de los pares, trazo por trazo, con la predicción de cada modelo |
| `fig_h4.png`, `fig_h5.png`, `fig_tsne.png` | Ver `06_...` y `07_...` |

## Tablas para el paper (`tablas.py`)

Al final de cada evaluación se escriben en `resultados/<particion>/tablas/` tres tablas en LaTeX: `tabla_busqueda.tex` (Tabla 1), `tabla_modelos.tex` (Tabla 2, media ± desviación de las semillas) y `tabla_hipotesis.tex` (el valor que decide cada criterio, por semilla). La plantilla del paper las incluye con `\input`, así que los números nunca se copian a mano. Para regenerarlas sin evaluar de nuevo: `tablas.desde_resultados(cfg.ruta_resultados, "validacion")`.

## Dónde modificar

- Otras fracciones: `FRACCIONES` en `config.py` (la AUC se normaliza sola con el primer y último valor).
- Otro umbral de trazos: `Config.min_trazos_progresivo`.
- Estilo de las figuras: `matplotlib.rcParams` y `COLORES` al inicio de `figuras.py`. Las figuras se rehacen ejecutando de nuevo la sección 5 del notebook, sin reentrenar.
