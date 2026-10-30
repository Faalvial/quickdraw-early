# 03 · Modelos M1, M2 y M2-P

**Archivo:** `quickdraw/modelos.py`
**Responsable en el plan:** Cristian
**Teoría:** secciones 3.3 (softmax, entropía cruzada, M1) y 3.4 (redes convolucionales, GAP, M2 y M2-P) del avance

## Qué hace

| Modelo | Clase | Arquitectura | Parámetros |
|---|---|---|---|
| M1 | `MLP` | 784 → 512 → 256 → 30, ReLU y dropout 0,3 | ~0,54 M |
| M2 | `CNN` | 3 bloques [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2] de 32, 64 y 128 filtros → GAP → dropout → Linear(128 → 30) | ~0,10 M |
| M2-P | `CNN` | La misma de M2 | La misma |

Dimensiones de la CNN: 1×28×28 → 32×14×14 → 64×7×7 → 128×3×3 → GAP → 128 → 30. La capa final tiene 128 × 30 + 30 = 3 870 parámetros (lo comprueba `tests/test_modelos.py`).

- `CNN.vectores(x)` devuelve el vector de 128 valores que sale del GAP. Es el espacio donde se mide la confundibilidad de H4 y se hacen los mapas t-SNE.
- `preparar_entrada` pasa las imágenes `uint8` de la caché a `float` entre 0 y 1 con forma N×1×28×28.
- Los modelos devuelven *logits*; el softmax lo aplica la entropía cruzada al entrenar y `torch.softmax` en la demo.

## Decisiones y por qué

- **M2 y M2-P son exactamente la misma red.** Lo único que cambia es con qué imágenes se entrena (ver `04_entrenamiento_y_busqueda.md`). Así H2 aísla el efecto de entrenar con prefijos.
- **M1 como referencia.** Responde "¿hace falta una CNN?": la MLP ve los mismos píxeles pero sin conectividad local ni parámetros compartidos.
- **GAP en vez de aplanar.** Con GAP la capa final tiene 3 870 parámetros en vez de 128·3·3·30 + 30 = 34 590, y el vector de 128 valores resume "qué patrones hay" sin importar dónde. Además permite cambiar la resolución sin tocar la red.
- **BatchNorm sin sesgo en la convolución.** BatchNorm ya resta la media, así que el sesgo de la convolución sería redundante (`bias=False`).
- **M0 (clasificador lineal) no se entrena.** Salió del alcance en el plan v5; queda en el marco teórico del avance.

## Dónde modificar

| Quieren… | Cambien… |
|---|---|
| Más o menos filtros | `CNN(filtros=(32, 64, 128))`. Si cambian el último, la capa final y los vectores de H4 cambian de tamaño solos. |
| Un cuarto bloque | `filtros=(32, 64, 128, 256)`. Con 28×28 quedaría 256×1×1 al final, todavía válido. |
| Otro dropout de M1 | `Config.m1_dropout` |
| Un modelo nuevo (por ejemplo M3) | Una clase aquí, una rama en `crear_modelo` y su nombre en `entrenar_modelos(..., nombres=...)`. Las métricas y figuras aceptan cualquier nombre. |

Después de cambiar la arquitectura hay que borrar `trabajo/modelos/` y `trabajo/resultados/predicciones_*`: este cambio **no** se detecta solo.
