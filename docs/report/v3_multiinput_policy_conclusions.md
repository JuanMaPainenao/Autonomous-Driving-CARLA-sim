# Reporte de Entrenamiento: PPO + MultiInputPolicy v3

## Resumen Ejecutivo

| Métrica | v2 (500k steps) | v3 (1.2M steps) |
|---|---|---|
| **Pico ep_rew_mean** | ~600–700 | ~104 |
| **Total timesteps** | 500.000 | 1.200.000 |
| **Reward type** | Aditiva | Multiplicativa |
| **Vector obs size** | 9 | 10 |
| **Resultado** | Plateau con exploits | **Regresión severa** |

El modelo v3 mostró una caída de ~6× en recompensa respecto a v2 a pesar de entrenar 2.4× más steps. La causa principal fue el cambio a reward multiplicativa, que transformó el entorno en un problema de reward efectivamente sparse para un agente no entrenado.

---

## Arquitectura de Red Neural

### MultiInputPolicy + CombinedExtractor

La política utiliza `MultiInputPolicy` de Stable Baselines3, que internamente instancia un `CombinedExtractor` como feature extractor. Este procesa las dos ramas de la observación `Dict`:

**Rama imagen** → NatureCNN (Mnih et al., 2015):

```
Input: (120, 160, 3) RGB
  → Conv2d(3, 32, kernel=8, stride=4) + ReLU   → (29, 39, 32)
  → Conv2d(32, 64, kernel=4, stride=2) + ReLU  → (13, 18, 64)
  → Conv2d(64, 64, kernel=3, stride=1) + ReLU  → (11, 16, 64)
  → Flatten → Linear(11264, 256)
Output: vector latente de 256 dimensiones
```

**Rama vector** → Flatten directo:

```
Input: (10,) float32
Output: (10,) sin transformación
```

**Fusión y cabezas:**

```
Concatenación: [256 (CNN) + 10 (vector)] = 266
  → Shared MLP: Linear(266, 64) + ReLU → Linear(64, 64) + ReLU
  → Actor head: Linear(64, 9)     → distribución Categorical (9 acciones discretas)
  → Critic head: Linear(64, 1)    → estimación de valor V(s)
```

Actor y Critic comparten el feature extractor (CNN + flatten) en PPO/A2C de SB3, lo cual reduce cómputo pero puede generar conflictos de gradiente entre la loss de política y la loss de valor.

### Ventajas de NatureCNN + CombinedExtractor

| Ventaja | Detalle |
|---|---|
| Simplicidad | No requiere configuración custom; SB3 lo instancia automáticamente |
| Probada en RL | NatureCNN fue diseñada para Atari (Mnih 2015) y es el estándar de facto |
| Multimodal | CombinedExtractor permite combinar imagen + datos numéricos sin código extra |
| Eficiencia | CNN compartida entre actor y critic reduce parámetros y VRAM |

### Desventajas de NatureCNN + CombinedExtractor

| Desventaja | Detalle |
|---|---|
| Diseñada para Atari (84×84 grayscale) | Las imágenes de CARLA (160×120 RGB) tienen una distribución visual muy diferente: texturas complejas, iluminación variable, perspectiva 3D |
| Sin normalización batch/layer | NatureCNN no usa BatchNorm ni LayerNorm, lo que puede hacer el training inestable con imágenes de alta varianza |
| Feature extractor compartido | El gradiente del critic puede interferir con el del actor, desestabilizando la política |
| Vector aplanado sin MLP | La rama vectorial (10 valores) se concatena cruda sin capas intermedias, lo que puede diluir su influencia respecto a los 256 features de la CNN |
| Sin memory temporal | NatureCNN procesa frames individuales; no captura dinámica temporal (aceleración, trayectoria reciente) |

---

## Hiperparámetros v3

| Parámetro | Valor | Justificación |
|---|---|---|
| learning_rate | linear_schedule(3e-4) | Decay lineal, parámetros Atari (CaRL 2025) |
| n_steps | 1024 | Rollouts más largos que v2 (512) |
| batch_size | 128 | Mini-batches más grandes, menos ruido |
| n_epochs | 4 | Reducido de 10; menos error off-policy (CaRL) |
| gamma | 0.99 | Estándar para tareas de horizonte medio |
| gae_lambda | 0.95 | Balance bias/varianza en GAE |
| clip_range | 0.2 | Estándar PPO |
| ent_coef | 0.005 | Reducido de 0.01; menos exploración forzada |
| target_kl | 0.015 | Más restrictivo que v2 (0.2); early stopping agresivo |
| max_grad_norm | 0.5 | Gradient clipping estándar |

---

## Función de Recompensa v3

### Estructura: base multiplicativa + penalidades aditivas

```
r_total = r_base + r_progress + r_steer_cost
        + r_wrong_way + r_stall + r_lane
        + r_collision + r_offroad + r_overspeed
```

Donde la base multiplicativa es:

```
r_base = 2.0 × speed_factor × centering × angle_factor
```

Cada factor está en [0, 1]. Si **cualquiera** es 0, toda la base es 0.

| Factor | Fórmula | Rango |
|---|---|---|
| speed_factor | exp(-(speed - 30)² / 100) | [0, 1] |
| centering | max(0, 1 - dist_center / lane_half_width) | [0, 1] |
| angle_factor | max(0, dot(veh_fwd, wp_fwd)) | [0, 1] |

### Componentes adicionales

| Componente | Valor | Tipo |
|---|---|---|
| r_progress | +0.1/metro (si dot > 0.5) | Continuo |
| r_steer_cost | -0.3 × \|Δsteer\| | Continuo |
| r_wrong_way | -2.0/step | Estado |
| r_stall | -0.1 a -3.0/step (progresivo) | Estado |
| r_lane | -2.0 | Evento |
| r_collision | -10.0 (termina episodio) | Evento |
| r_offroad | -3.0/step | Estado |
| r_overspeed | -1.0/step (>50 km/h) | Estado |

---

## Diagnóstico: Por qué v3 alcanzó solo 104 de pico

### 1. Reward multiplicativa genera reward efectivamente sparse

El problema central es que la base multiplicativa `speed × centering × angle` requiere que **los tres factores sean simultáneamente altos** para dar reward significativa. Un agente aleatorio al inicio del entrenamiento:

- Va a velocidad incorrecta → speed_factor ≈ 0.3
- Está descentrado → centering ≈ 0.4
- Mal alineado → angle_factor ≈ 0.5

Producto: 2.0 × 0.3 × 0.4 × 0.5 = **0.12** por step.

Comparado con v2 donde r_speed + r_orientation sumaban ~1.0–1.5 por step para el mismo comportamiento. La señal de gradiente es ~10× más débil desde el inicio.

Esto crea un **problema de sparse reward**: el agente recibe tan poca reward positiva que no puede distinguir "buena acción" de "mala acción", y PPO no tiene suficiente señal para mejorar la política. La literatura (Pathak 2017, Burda 2018) confirma que PPO es particularmente vulnerable a rewards sparse porque es on-policy y descarta los datos después de cada update.

### 2. Steering cost penaliza la exploración temprana

El steering cost (-0.3 × |Δsteer|) es correcto en principio (Think2Drive lo usa), pero perjudica al agente no entrenado. Un agente random zigzaguea por naturaleza, acumulando -0.09 a -0.18 por step extra. Combinado con la base multiplicativa baja, la reward total por step se acerca a 0 o se vuelve negativa, eliminando la señal de aprendizaje.

### 3. σ más estrecha en speed_factor

v2 usaba σ=20 en la gaussiana de velocidad: `exp(-(Δspeed/20)²)`. v3 la estrechó a σ=10: `exp(-(Δspeed/10)²)`. A 0 km/h (agente parado), v2 daba 0.32 vs v3 da 0.11. El agente arranca más penalizado antes de aprender a moverse.

### 4. MAX_STEPS = 4000 amplifica el problema

Con episodios de 4000 steps y reward cercana a 0 por step, `ep_rew_mean` acumula muchos steps de reward mínima. Mientras que v2 con MAX_STEPS=2000 y reward aditiva de ~1.5/step daba ~3000 de reward teórica máxima, v3 con ~0.1/step de base da ~400 teóricos si todo sale perfecto.

### 5. Resumen cuantitativo

| Escenario | Reward/step v2 | Reward/step v3 |
|---|---|---|
| Conducción perfecta (30 km/h, centrado, alineado) | +2.0 | +2.0 + 0.1 = +2.1 |
| Conducción aceptable (20 km/h, algo descentrado) | +1.5 | +0.8 |
| Agente aleatorio | +0.5 a +1.0 | +0.05 a +0.15 |
| Parado | -0.1 | -0.1 |

La diferencia crítica está en el **agente aleatorio**: v2 le da ~10× más señal que v3, permitiendo que PPO identifique las acciones ligeramente mejores y construya gradiente desde ahí.

---

## Conclusiones

1. **La reward multiplicativa no es inherentemente mala** — CaRL (2025) y Roach la usan con éxito — pero requiere que el agente ya tenga una política base razonable. Introducirla desde el step 0 con un agente aleatorio genera un problema de reward sparse.

2. **La reward aditiva de v2 era más "friendly" para aprendizaje from scratch** porque cada componente daba señal independiente. El agente podía aprender "ir rápido" y "alinearme" como objetivos separados, y después combinarlos.

3. **El steering cost debería activarse en una fase tardía** (curriculum learning) o con un peso mucho menor al inicio (ej: 0.05 en vez de 0.3).

4. **Los hiperparámetros del PPO (lr schedule, n_epochs=4, target_kl=0.015) son correctos** según la literatura y no fueron la causa del bajo rendimiento.

---

## Recomendaciones para v4

| Cambio | Razón |
|---|---|
| Volver a reward aditiva para la base (v2 style) | Garantiza señal densa desde el step 0 |
| Agregar centering como componente aditivo separado (+0.5 × centering) | Incentiva centrado sin depender del producto |
| Steering cost con peso 0.05 (no 0.3) o activar después de 200k steps | No castigar exploración temprana |
| Mantener lr schedule lineal, n_epochs=4, target_kl=0.015 | Estos cambios son positivos |
| Mantener r_progress (+0.1/m) condicionado a dot > 0.5 | Incentiva avance sin premiar contramano |
| Considerar σ=15 en speed_factor (intermedio entre v2 y v3) | Más tolerante al inicio, más exigente que v2 |

---

## Referencias

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Vergara (2019). *Accelerating Training of DRL-based Autonomous Driving Agents* (bitsauce/Carla-ppo).
- Li et al. (2024). *Think2Drive: Efficient RL by Thinking in Latent World Model*. ECCV 2024.
- Jaeger et al. (2025). *CaRL: Learning Scalable Planning Policies with Simple Rewards*. CoRL 2025.
- Stable Baselines3 Documentation. *Policy Networks & Custom Policies*.
