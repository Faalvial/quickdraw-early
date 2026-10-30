# 01 · Datos y particiones

**Archivos:** `quickdraw/datos.py`, `quickdraw/duracion.py`, `datos/download_sample.py`
**Responsable en el plan:** Elias
**Teoría:** sección 3.1 del avance (datos y definición de trazo) y sección 3.6.1 (entrenamiento, validación y prueba)

## Qué hace

1. `download_sample.py` recorre el archivo *simplified* completo de cada categoría y toma una muestra uniforme con muestreo por reservorio (algoritmo R): 10 000 dibujos reconocidos y 1 000 no reconocidos, con la semilla 2026. Ya está hecho; el resultado es `muestra_30/`.
2. `cargar_muestra` lee los primeros `dibujos_por_categoria` dibujos reconocidos de cada archivo. Cada dibujo queda como un diccionario con `key_id`, `categoria`, `etiqueta` (0 a 29), `pais` y `trazos` (lista de pares `[xs, ys]`).
3. `particionar` divide 70/15/15 **por dibujo**, con la misma proporción en cada categoría (estratificado) y la semilla 2026.
4. `guardar_particiones` escribe los `key_id` de cada partición, para que cualquiera pueda reproducir la división.
5. `tabla_hechos` resume cada categoría: dibujos, mediana de trazos, cuántos tienen menos de 3 trazos y país más frecuente.
6. `duracion.py` (opcional) mide la duración de los dibujos de "cat" en el formato *raw*.

## Decisiones y por qué

- **Solo dibujos reconocidos.** Son los que el juego aceptó como la categoría: la etiqueta es más confiable. Los no reconocidos se guardaron en la muestra pero el proyecto no los usa.
- **Partir por dibujo y antes de generar prefijos.** Si se partiera por prefijo, el prefijo 3 de un dibujo podría quedar en entrenamiento y el prefijo 4 del mismo dibujo en prueba, y el modelo "recordaría" el dibujo. Partiendo antes, todos los prefijos de un dibujo quedan juntos. Esto es lo que se defiende en la pregunta de defensa sobre fuga de datos.
- **Estratificado.** Cada categoría tiene exactamente 7 000 / 1 500 / 1 500 dibujos, así que la exactitud no depende de qué categorías tuvieron más suerte en el sorteo.
- **Langosta en vez de trompeta.** Trompeta tenía una mediana de 4 trazos; con tan pocos trazos casi no hay curva progresiva. Langosta fue la siguiente categoría que salió con la misma semilla.
- **Duración (premisa de H5).** La ronda del juego termina cuando la IA acierta. Si los dibujos reconocidos duran bastante menos que los 20 s del límite, eso apoya que el juego cortó al dibujante antes de que terminara. El formato *simplified* no tiene tiempos; por eso se usa el *raw*, solo en "cat" porque cada archivo pesa cientos de MB.

## Dónde modificar

| Quieren… | Cambien… |
|---|---|
| Menos dibujos por categoría (para ir más rápido) | `Config.dibujos_por_categoria`. La caché se rehace sola; los modelos hay que borrarlos (ver `00_como_ejecutar.md`). |
| Otras proporciones | `Config.proporciones` |
| Otras categorías | `CATEGORIAS_PARES` / `CATEGORIAS_AZAR` en `config.py`, y la lista en `download_sample.py`. Hay que volver a descargar. |
| Medir otra categoría en *raw* | `medir_duracion("dog")` en la sección 8 del notebook |

## Qué revisar a mano

- `tabla_hechos.csv`: que cada categoría tenga 10 000 dibujos y que las medianas de trazos coincidan con las del avance.
- `particiones/key_ids_*.json`: que no haya un `key_id` repetido entre archivos (la prueba `tests/test_particion.py` lo verifica con datos sintéticos).
