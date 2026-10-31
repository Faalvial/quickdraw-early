# 04 · Entrenamiento y búsqueda de hiperparámetros

**Archivos:** `quickdraw/entrenar.py`, `quickdraw/experimentos.py` (etapas 2 y 3)
**Responsable en el plan:** Cristian
**Teoría:** sección 3.3.2 (entropía cruzada), 3.4.6 (entrenamiento con dibujos parciales) y 3.6.1 (validación) del avance; sección 3 del plan v5

## Qué hace

### Qué imagen ve cada modelo (`indices_de_epoca`)

- **M1 y M2 (`'completos'`):** en cada época, el dibujo completo de cada dibujo de entrenamiento.
- **M2-P (`'prefijos'`):** en cada época, **un prefijo al azar** de cada dibujo, con k uniforme entre 1 y K. En la época siguiente se sortea otro k.
- **M2-P (`'mixto'`, el adoptado):** igual, pero con probabilidad `m2p_prob_completo` = 0,5 se usa el dibujo completo en lugar del prefijo. Ver el final de este documento.

Cada dibujo aporta una imagen por época en todos los esquemas, así que M2 y M2-P hacen el mismo número de pasos y los dibujos con muchos trazos no pesan más.

### Bucle de entrenamiento (`entrenar`)

- Entropía cruzada, lotes de 256, calendario de tasa de aprendizaje coseno (baja suavemente hasta 0 en `epocas_max`).
- Adam, o SGD con momentum 0,9 (Nesterov).
- Precisión mixta (AMP) cuando hay GPU: casi el doble de rápido en una T4 sin cambiar los resultados de forma apreciable.
- **Parada temprana:** si el objetivo de validación no mejora en 3 épocas (`paciencia`), se detiene y se queda con los pesos de la mejor época.
  - M1 y M2: exactitud con el dibujo completo en validación.
  - M2-P: AUC progresiva en una submuestra fija de 3 000 dibujos de validación.
- Un `checkpoint.pt` por época, para retomar si Colab corta.

### Búsqueda (`busqueda`, `rejilla_m2`, `rejilla_m1`)

- M2: {Adam, SGD} × tasa {0,003, 0,001, 0,0003} × dropout {0, 0,3} = 12 combinaciones, con 5 épocas y el 30 % del entrenamiento.
- M1: las 3 tasas con Adam y dropout 0,3.
- Se elige la combinación con mejor exactitud de validación. **M2-P hereda la de M2.**
- Resultado: `tabla_busqueda.csv` (una tabla del paper) y `hiperparametros_elegidos.json`.

### Semillas

M2 y M2-P se entrenan con las semillas 0, 1 y 2; M1 solo con la 0 (`semillas_m1`). La semilla fija la inicialización, el orden de los lotes y los prefijos sorteados. Las hipótesis se confirman solo si se cumplen en las tres.

## Decisiones y por qué

- **M2-P hereda los hiperparámetros de M2.** Si cada uno tuviera su propia búsqueda, H2 compararía dos cosas a la vez (datos y configuración). Así cambia una sola.
- **La búsqueda es corta (5 épocas, 30 %).** Sirve para ordenar las combinaciones, no para el resultado final. Es la respuesta a "¿por qué esos valores?" en la exposición.
- **Parada temprana de M2-P con la AUC.** *Esto difiere un poco del plan v5,* que dice "exactitud de validación" para todos. M2-P se entrena para acertar temprano; si se detuviera mirando solo el dibujo completo, podría quedarse con una época que acierta al final pero no antes. Si el grupo prefiere seguir el plan al pie de la letra, en `entrenar_o_cargar` cambien `(cfg.esquema_m2p, "auc")` por `(cfg.esquema_m2p, "final")` y reentrenen M2-P.
- **Calendario coseno y AMP.** No estaban en el plan; son prácticas estándar que estabilizan el final del entrenamiento y ahorran GPU. Para quitarlos, borren `sched.step()` o pongan `usar_amp = False`.
- **"Quien entrena no evalúa".** Este módulo solo entrena y predice; todas las métricas salen de `metricas.py`.

## Dónde modificar

| Quieren… | Cambien… |
|---|---|
| Otra rejilla | `busqueda_optimizadores`, `busqueda_lr`, `busqueda_dropout` en `Config` |
| Más épocas o más paciencia | `epocas_max`, `paciencia` |
| M1 con 3 semillas (si sobra tiempo) | `semillas_m1=(0, 1, 2)` |
| Entrenar solo un modelo | `E.entrenar_modelos(cfg, D, hp, nombres=("M2-P",))` |
| Elegir hiperparámetros a mano | Editen el diccionario `hp` en el notebook antes de `entrenar_modelos` |

### Esquema de M2-P: por qué se adoptó el mixto

**Esquema adoptado: mixto (31/10).** Con el esquema original de prefijos, la corrida oficial en validación (3 semillas) dio a M2-P una ventaja de AUC de +0,070 a +0,096, pero perdió 1,61, 2,99 y 0,87 puntos de exactitud final: H2 quedaba refutada porque el criterio permite 1. Se aplicó el plan B escrito de antemano: **mezclar dibujos completos**, con probabilidad 0,5 el dibujo completo y si no un prefijo al azar (`esquema_m2p="mixto"`, `m2p_prob_completo=0.5`). En validación, la ventaja de AUC quedó entre +0,058 y +0,088 y la pérdida en 0,66, 0,94 y −1,01 puntos: H2 se cumple en las tres semillas. La decisión se tomó con validación, antes de abrir la prueba, y el paper reporta los dos esquemas.

Desde entonces `"mixto"` es el valor por defecto en `config.py`. Para volver a evaluar el esquema original (por ejemplo, la carpeta `trabajo_prefijos/`), agreguen al crear `cfg` el argumento `esquema_m2p="prefijos"`. Si cambian el esquema o la proporción, borren antes `modelos/M2-P_s*` y `resultados/predicciones_*` de esa carpeta, o el notebook se detiene avisando que la configuración cambió.

Es un cambio de método: se decidió con los resultados de **validación** (nunca de prueba) y se declara en el paper.
