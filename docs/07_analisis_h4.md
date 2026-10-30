# 07 · Análisis por categoría: H4, t-SNE y pares

**Archivo:** `quickdraw/analisis.py` (y `hipotesis.h4`)
**Responsable en el plan:** Farid
**Teoría:** secciones 3.4.4 (GAP), 3.5.2 y 3.5.3 (t-SNE y su uso en el proyecto) del avance; "H4 reformulada" en la sección 1 del plan v5

## Qué hace

### Confundibilidad (`vectores_cnn`, `confundibilidad`)

1. Se toma M2 (entrenada con dibujos completos) de cada semilla.
2. Se extrae el vector GAP de 128 valores de cada dibujo **completo de validación**.
3. Para cada dibujo se buscan sus 10 vecinos más cercanos (distancia euclidiana, sin contarse a sí mismo).
4. Pureza de un dibujo = proporción de vecinos de su misma categoría. **Confundibilidad** de una categoría = 1 − pureza promedio de sus dibujos.

### Trazos necesarios (`mediana_fraccion_estrella`)

Mediana de k*/K de M2-P por categoría en la partición que se evalúa. Los dibujos que M2-P nunca reconoce cuentan como 1.

### H4

Spearman entre las 30 confundibilidades y las 30 medianas, en cada semilla. `fig_h4.png` muestra la dispersión con el nombre de cada categoría (semilla 0).

### t-SNE (`mapa_tsne`), solo para mirar

Dos mapas de los vectores GAP con 150 dibujos por categoría: uno con el 30 % del dibujo y otro con el 100 %. Los 10 de los pares van en color y las 20 al azar en gris. Deberían verse los pares mezclados al 30 % y separados al 100 %. Ninguna conclusión se apoya en el mapa: t-SNE no conserva distancias globales ni tamaños de grupo, y depende de la perplejidad.

### Separación de pares (`separacion_de_pares`)

Para cada par (gato/perro, león/tigre, etc.), qué porcentaje de sus dibujos se confunde con la otra categoría del par en cada fracción. Responde "¿en qué trazo se separan?". Resultado: `separacion_pares.csv`.

## Decisiones y por qué

- **Espacio de la CNN y no PCA.** Por la observación del profesor: PCA sobre píxeles mide parecido de tinta, no lo que la red aprendió. El vector GAP es lo que la última capa usa para decidir.
- **No es circular.** La confundibilidad sale de M2 con dibujos completos; los trazos necesarios salen de M2-P, otra red entrenada de otra forma, y con prefijos. Sí depende de una CNN, y eso se declara como limitación.
- **kNN en vez de distancia a centroides.** No supone que cada categoría forme una nube redonda.
- **Spearman.** Solo pide que las dos medidas crezcan juntas, no que lo hagan en línea recta, y con 30 puntos es menos sensible a una categoría extrema.
- **t-SNE con M2-P.** *Esto difiere un poco del plan v5,* que dice "esos mismos vectores" (de M2). M2 nunca vio prefijos, así que su mapa al 30 % mostraría sobre todo imágenes que no sabe representar; M2-P sí se entrenó con prefijos y su mapa muestra cómo se van separando las categorías. Para usar M2, en `experimentos.evaluar` cambien `modelos[("M2-P", s0)]` por `modelos[("M2", s0)]` en el bloque de `mapas`.

## Dónde modificar

| Quieren… | Cambien… |
|---|---|
| Otro número de vecinos | `Config.vecinos_knn` (por ejemplo 5 o 20, como análisis de sensibilidad) |
| Mapas con más puntos o distinta perplejidad | `Config.tsne_por_categoria`, `Config.tsne_perplejidad` |
| Mapas en otra fracción (por ejemplo 50 %) | El bucle `for f in (0.3, 1.0)` en `experimentos.evaluar` |
| Matrices de confusión de M2 además de M2-P | Llamen `figuras.confusion(metricas.matriz_confusion(cache, pred_m2, f), ...)` con las predicciones de M2 |

Si el grupo se atrasa, el plan permite que H4 quede solo como análisis visual (t-SNE y matrices de confusión): basta con no reportar el veredicto de H4.
