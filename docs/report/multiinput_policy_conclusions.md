# Reward Model Comparison: v1 vs v2

## Overview

| | v1 | v2 |
|---|---|---|
| **Key feature** | progress reward + hard stall | wrong_way penalty + progressive stall |
| **Vector obs size** | 7 | 9 (+wrong_way, +vel_dot) |
| **Checkpoint compat** | — | **BREAKING**: v1 checkpoints won't load |

---

## Vector Observation

| Index | v1 | v2 | Change |
|---|---|---|---|
| 0 | speed_norm | speed_norm | — |
| 1 | collision | collision | — |
| 2 | orientation | orientation | — |
| 3 | dist_to_center | dist_to_center | — |
| 4 | angle_diff | angle_diff | — |
| 5 | is_offroad | is_offroad | — |
| 6 | stall_ratio (/30) | stall_ratio (/200) | rescaled |
| 7 | — | **wrong_way** (0 or 1) | **new** |
| 8 | — | **vel_dot** [-1, 1] | **new** |

**wrong_way** gives the agent an explicit signal it's driving against traffic. Without it, the CNN would have to infer this from the image alone — nearly impossible on straight roads where both lanes look identical.

**vel_dot** is the dot product between the vehicle's velocity vector and the waypoint forward direction. Provides a continuous measure of how aligned actual movement is with the legal lane flow.

---

## Reward Components

### v1

| Component | Value | Trigger |
|---|---|---|
| r_speed | +1.0 max (gaussian) | At 30 km/h |
| r_speed (stalled) | **-5.0** | speed < 2 km/h |
| r_orientation | +1.0 max | dot > 0, speed > 5 km/h |
| r_progress | +1.0 × distance | Euclidean dist between steps |
| r_lane | -5.0 | Lane line crossed (event) |
| r_collision | -10.0 | Collision (terminates episode) |
| r_offroad | -5.0 | Outside Driving lane |
| r_stall | -3.0 + termination | stall ≥ 30 steps (~1.5s) |

### v2

| Component | Value | Trigger |
|---|---|---|
| r_speed | +1.0 max (gaussian) | At 30 km/h |
| r_orientation | +1.0 max | dot > 0, speed > 5 km/h |
| **r_wrong_way** | **-2.0/step** | Moving against lane direction |
| **r_stall** (grace) | **-0.1/step** | Stopped < 2s (40 steps) |
| **r_stall** (ramp) | **-0.5 to -3.0/step** | Stopped > 2s, scales up |
| **r_stall** (terminate) | -3.0 + end episode | Stopped > 10s (200 steps) |
| r_lane | -5.0 | Lane line crossed (event) |
| r_collision | -10.0 | Collision (terminates episode) |
| r_offroad | -5.0 | Outside Driving lane |

---

## What Changed and Why

### 1. Removed r_progress

`r_progress = 1.0 × distance` rewarded movement in any direction, including wrong way. Speed is already rewarded by r_speed; direction by r_orientation.

### 2. Added r_wrong_way (-2.0/step)

In v1, swerving into oncoming traffic was cheaper than braking. The lane invasion penalty (-5.0) fired once when crossing the line; after that, wrong-way driving was free. Now every step in the wrong lane costs -2.0, detected via velocity vector vs waypoint direction (dot < 0 → wrong way).

**Per-step economics:**

| Situation | Reward/step |
|---|---|
| Driving correctly at 30 km/h | **+2.0** |
| Wrong way at 30 km/h | **~0.0** |
| Braking behind a car (< 2s) | **-0.1** |

### 3. Progressive stall replaces hard cutoff

v1 terminated after 30 steps (~1.5s) stopped with -5.0/step. Forced the agent to swerve instead of waiting.

v2 three phases:
- **Grace (0–40 steps, ~2s):** -0.1/step. Total cost ~-4.0
- **Ramp (40–200 steps):** -0.5 to -3.0/step, scales gradually
- **Terminate (>200 steps, ~10s):** Episode ends

### 4. Removed flat -5.0 for speed < 2 km/h

One braking step cost as much as a lane invasion. The agent never wanted to brake. Now the gaussian gives ~0.37 at 0 km/h (low but not catastrophic), and stall penalty builds gradually.

---

## TensorBoard: What to Look For

### Working well
- `ep_rew_mean` rises above v1 plateau (~200)
- `ep_len_mean` increases
- CSV: `avg_r_wrong_way` near 0
- CSV: `avg_r_stall` between -0.1 and 0

### Trouble signs and fixes

| Symptom | Fix |
|---|---|
| Still going wrong way | Increase R_WRONG_WAY to -3.0 |
| Never brakes | Decrease STALL_SOFT_PENALTY to -0.05 |
| Stays parked | Increase STALL_RAMP to -0.05 |
| Episodes too short | Increase STALL_TERMINATE_STEPS to 300 |

---

## How to Run

```bash
# v1 checkpoints are incompatible — must use --fresh
python3.10 ppo_carla_train.py --fresh --preview
```