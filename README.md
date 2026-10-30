# ¿Cuántos trazos bastan? Reconocimiento temprano de bocetos con CNN en Quick, Draw!

Proyecto final de CC421 Inteligencia Artificial, UNI, 2026-II.

**Pregunta.** ¿Con cuántos trazos reconoce una CNN un dibujo, y por qué algunas categorías necesitan más?

Se entrenan tres modelos sobre 30 categorías de Quick, Draw! (5 pares confundibles y 20 al azar; 10 000 dibujos reconocidos por categoría) y se evalúan con cada prefijo de cada dibujo:

- **M1**, perceptrón multicapa, como referencia;
- **M2**, CNN entrenada con dibujos completos;
- **M2-P**, la misma CNN entrenada con un prefijo al azar de cada dibujo.

## Empezar

Lean [`docs/00_como_ejecutar.md`](docs/00_como_ejecutar.md). En resumen: suban esta carpeta y `muestra_30` a `MyDrive/quickdraw-early/`, abran `notebooks/proyecto_quickdraw.ipynb` en Colab con GPU y ejecuten todo; primero con `MODO_RAPIDO = True` para comprobar que corre.

## Estructura

| Ruta | Qué hay | Documento | Responsable |
|---|---|---|---|
| `notebooks/proyecto_quickdraw.ipynb` | Todo el proyecto, en orden, con "Ejecutar todas" | [00](docs/00_como_ejecutar.md) | Elias |
| `quickdraw/config.py` | Todos los parámetros en un solo lugar | [00](docs/00_como_ejecutar.md) | — |
| `quickdraw/datos.py`, `quickdraw/duracion.py`, `datos/download_sample.py` | Muestra, particiones por dibujo, tabla de hechos, duración | [01](docs/01_datos.md) | Elias |
| `quickdraw/raster.py` | Prefijos, normalización causal, rasterización y caché | [02](docs/02_prefijos_y_rasterizacion.md) | Jorly |
| `quickdraw/modelos.py` | M1, M2 y M2-P | [03](docs/03_modelos.md) | Cristian |
| `quickdraw/entrenar.py` | Entrenamiento, checkpoints y búsqueda de hiperparámetros | [04](docs/04_entrenamiento_y_busqueda.md) | Cristian |
| `quickdraw/metricas.py`, `quickdraw/figuras.py`, `quickdraw/tablas.py` | Exactitud por fracción, AUC progresiva, k*, confusión, figuras y tablas LaTeX | [05](docs/05_metricas.md) | Omar |
| `quickdraw/estadistica.py`, `quickdraw/hipotesis.py` | Bootstrap, McNemar, Spearman y criterios de H1, H2, H4 y H5 | [06](docs/06_estadistica_e_hipotesis.md) | Omar |
| `quickdraw/analisis.py` | Confundibilidad en el espacio de la CNN, t-SNE y pares | [07](docs/07_analisis_h4.md) | Farid |
| `demo/demo.py`, `notebooks/demo.ipynb` | Demo en Gradio | [08](docs/08_demo.md) | Jorly |
| `quickdraw/experimentos.py` | Las etapas que llama el notebook | [00](docs/00_como_ejecutar.md) | — |
| `tests/` | Pruebas automáticas (`pytest tests`) | — | — |

## Hipótesis

| Hip. | Mitad de la pregunta | Se confirma si (en las 3 semillas) |
|---|---|---|
| H1 | ¿Con cuántos trazos? | M2 pierde más de 10 puntos de exactitud a mitad del dibujo, con McNemar p < 0,05 |
| H2 | ¿Con cuántos trazos? | M2-P tiene más AUC progresiva que M2 (IC 95 % > 0) sin perder más de 1 punto al final |
| H5 | ¿Con cuántos trazos? | M2-P acierta de forma estable antes del último trazo en más del 50 % de los dibujos |
| H4 | ¿Por qué algunas necesitan más? | Las categorías más confundibles en el espacio de M2 necesitan más trazos en M2-P (Spearman ρ > 0, p < 0,05) |

## Reproducibilidad

- Semilla de la muestra y de las particiones: 2026. Los `key_id` de cada partición se guardan en `trabajo/resultados/particiones/`.
- Semillas de entrenamiento: 0, 1 y 2.
- El conjunto de prueba se usa una sola vez (semana 13), cambiando `USAR_PRUEBA` a `True`.

## Uso de IA

El código de este repositorio fue generado con asistencia de IA (Claude) a partir del plan del grupo, y revisado y modificado por el grupo. Debe declararse en la DUIA del entregable final.
