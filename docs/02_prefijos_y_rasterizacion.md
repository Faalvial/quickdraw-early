# 02 · Prefijos, normalización causal y rasterización

**Archivo:** `quickdraw/raster.py`
**Responsable en el plan:** Jorly
**Teoría:** sección 3.2 del avance (representación de los bocetos: trazos, normalización, rasterización, imágenes progresivas y causalidad)

## Qué hace

Para un dibujo con K trazos genera K imágenes de 28×28: la imagen k tiene solo los primeros k trazos (el **prefijo** k).

1. `normalizar(trazos, cfg)` lleva las coordenadas al lienzo: resta la esquina de la caja delimitadora, escala con **un solo factor** a = (L − 1 − 2m) / max(ancho, alto) y centra. Es la ecuación de normalización del avance.
2. `rasterizar` dibuja las líneas en un lienzo de 256×256 con grosor 16 y extremos redondeados, y lo reduce a 28×28 promediando bloques (`Image.BOX`). Dibujar en grande y reducir suaviza los bordes, como en las imágenes de 28×28 que publica Google.
3. `prefijos_de_dibujo` aplica lo anterior a cada prefijo.
4. `construir_cache` rasteriza todos los prefijos de una partición en paralelo y los guarda seguidos en un solo arreglo `imagenes`. Con `inicio[i]` y `K[i]` se encuentra cada prefijo: el prefijo k del dibujo i está en `imagenes[inicio[i] + k - 1]`.
5. `k_de_fraccion(K, f) = max(1, ⌈f·K⌉)` define qué prefijo corresponde a cada fracción del dibujo.

## Decisiones y por qué

- **Normalización causal.** Cada prefijo se normaliza con la caja de **sus propios** trazos, no con la del dibujo completo. Así la imagen del prefijo k no depende de nada que el dibujante todavía no hizo, que es la situación real del juego. La caja final sería hacer trampa: le diría al modelo dónde va a terminar el dibujo. La figura `fig_prefijos.png` muestra las dos versiones (la fila "caja final" es solo de referencia).
- **Un solo factor de escala.** Con factores distintos para x e y, una línea horizontal se estiraría hasta ocupar un cuadrado y se perdería la forma.
- **Un solo punto.** Si el prefijo es un punto (caja de tamaño cero), se dibuja en el centro en vez de dividir entre cero.
- **Margen de 1 píxel.** Evita que la tinta toque el borde, donde las convoluciones con relleno ven ceros.
- **Caché en disco.** Rasterizar los ~2,2 millones de prefijos tarda varios minutos; se hace una vez y se guarda en Drive (unos 1,7 GB con prueba). El nombre del archivo lleva los parámetros, así que si los cambian se crea una caché nueva.
- **En memoria, sin DataLoader.** Todas las imágenes caben en la RAM de Colab como `uint8`, así que entrenar es solo indexar ese arreglo.

## Dónde modificar

| Quieren… | Cambien… |
|---|---|
| Trazo más fino o más grueso | `Config.grosor` (16 en el lienzo de 256 ≈ 1,75 px a 28×28) |
| Otra resolución (por ejemplo 32×32) | `Config.resolucion`. La CNN se adapta sola (usa GAP); la MLP también (`resolucion * resolucion` entradas). |
| Ver cómo cambia la imagen | `figuras.prefijos(...)` en el notebook, sección 2 |
| Comparar con la normalización no causal | Pasar `caja_ref=caja(trazos_completos)` a `rasterizar`. Solo para análisis; nunca para entrenar ni evaluar. |

Después de cambiar la rasterización hay que reentrenar: ver la tabla de `00_como_ejecutar.md`.

## Pruebas

`tests/test_raster.py` comprueba que agregar trazos futuros no cambia la imagen de un prefijo (causalidad), que un punto queda al centro, que se conserva la proporción, que el borde queda casi vacío y que `k_de_fraccion(7, 0.3) = 3`.
