# Reward Function Comparison — CarlaEnv

![Training metrics](../../images/Entrenamiento.png)

The right image shows episode duration over time. The left image shows episode reward over time.

## First Model — 5 Components

| Component | Value / Formula | Condition |
|---|---|---|
| **r_speed** | `1.0 × exp(-(speed - 40)² / 400)` | Always (Gaussian, never negative) |
| **r_orientation** | `1.0 × max(0, dot_product)` | Always (no speed threshold) |
| **r_lane** | `-5.0` | On lane invasion |
| **r_collision** | `-10.0` | On collision → episode ends |
| **r_offroad** | `-5.0` | If outside the Driving lane |

**Total formula:** `r_speed + r_orientation + r_lane + r_collision + r_offroad`

---

## Second Model — 7 Components

| Component | Value / Formula | Condition |
|---|---|---|
| **r_speed** | `1.0 × exp(-(speed - 40)² / 400)` | If `speed ≥ 2 km/h` (Gaussian) |
| **r_speed** (stall) | `-5.0` | If `speed < 2 km/h` |
| **r_orientation** | `1.0 × max(0, dot_product)` | Only if `speed > 5 km/h`, otherwise `0.0` |
| **r_progress** | `1.0 × min(distance_traveled, 5.0)` | Distance between frames, capped at 5m |
| **r_lane** | `-5.0` | On lane invasion |
| **r_collision** | `-10.0` | On collision → episode ends |
| **r_offroad** | `-5.0` | If outside the Driving lane |
| **r_stall** | `-3.0` | After ≥30 steps stopped → episode ends |

**Total formula:** `r_speed + r_orientation + r_progress + r_lane + r_collision + r_offroad + r_stall`

---

## Shared Constants

| Parameter | Value |
|---|---|
| TARGET_SPEED_KMH | 40 km/h |
| MAX_STEPS | 2000 (100s simulated) |
| FIXED_DELTA | 0.05s (20 simulated FPS) |
| Discrete actions | 9 combinations (steer ±0.3, throttle 0.6/1.0, brake 0.5) |

---

## Differences Between Versions

### 1. Stillness Penalty (r_speed when speed < 2 km/h)
Penalizes the agent with -5 if it remains stopped or moves slower than 2 km/h.

### 2. r_progress (V2 only)
Reward proportional to the actual distance traveled between frames. Incentivizes forward displacement.

### 3. Speed-gated r_orientation
Prevents the agent from receiving reward for "facing the right direction" without actually moving.

### 4. r_stall — Inactivity cutoff
If the agent accumulates ≥30 consecutive steps with `speed < 2 km/h`, the episode ends with an additional penalty of `-3.0`. This forces the agent to move or lose the episode.

### 5. CSV Logging
V2 logs `r_progress` and `r_stall` as additional columns in the CSV. V1 only records 5 components.

---

# Conclusions

Assuming MAX_STEPS = 2000 and the ideal case (no collision, no early truncation):

- **V1 — Theoretical maximum per episode:** `2.0 × 2000 = 4000`
- **V2 — Theoretical maximum per episode:** `7.0 × 2000 = 14000`
- **V2 — Realistic maximum per episode:** `2.56 × 2000 ≈ 5120`

> **Note on V2 realistic maximum:** this estimate assumes the agent drives consistently at or near the target speed (yielding ~1.0 from r_speed), travels ~1.56m per frame on average (r_progress), and maintains correct orientation while avoiding all penalties. The theoretical ceiling of 7.0 per step is unattainable in practice because r_progress is capped at 5.0 and requires continuous forward movement without any collisions, lane invasions, or stall events.

It is concluded that V1 is the weaker model with a poorly designed reward structure: by the end of training, the last episode ended with a negative reward. Observation in CARLA confirmed that the vehicle was standing still — the agent had learned that staying stopped was preferable to risking the collision penalty.

V2 is more robust; however, it is still well below the realistic maximum reward per episode, suggesting the model requires additional training.