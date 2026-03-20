# Comparative Autonomous Driving System

**Modular Pipeline, Reinforcement Learning, and Imitation Learning in CARLA with Cross-Validation in CoppeliaSim**

---

## Overview

This project develops, trains, and compares three autonomous driving approaches in the [CARLA](https://carla.org/) simulator:

- **Modular Pipeline** — classical perception + planning + control stack
- **Reinforcement Learning** — model-free agents (PPO, SAC) trained end-to-end
- **Imitation Learning** — Behavior Cloning (BC) and Conditional Imitation Learning (CIL) from expert demonstrations

All five resulting models are evaluated under identical conditions using standardized metrics, and then transferred to [CoppeliaSim](https://www.coppeliarobotics.com/) to analyze the sim-to-sim gap.

> **Academic context:** Final project (PPS) — FCEFyN, Universidad Nacional de Córdoba · February 2026

---

## Objectives

1. Implement a classical **Modular Pipeline** (perception, planning, control) as a baseline.
2. Train **Reinforcement Learning** agents in CARLA using PPO and SAC.
3. Train **Imitation Learning** models (BC and CIL) from driving demonstrations.
4. Design a **comparative evaluation** framework with standard metrics: route completion rate, infraction-free distance, training time, policy stability, and out-of-distribution behavior.
5. **Cross-validate** trained models in CoppeliaSim via its Python API.
6. Analyze the **inter-simulator gap** and draw conclusions on each method's viability.

---

## Project Structure

```
├── docs/                       # Administrative and theoretical documentation
│   ├── administration/         # Work plan, Gantt chart (PDF/JPG), FCEFyN forms
│   ├── theory/                 # Theoretical framework (.md)
│   └── report/                 # Final report and conclusions (.md)
├── src/                        # Source code
│   ├── agents/                 # Autonomous driving logic (AI policies)
│   ├── training/               # Neural network training scripts
│   ├── utils/                  # Helper scripts (image processing, sensors)
│   └── main.py                 # Entry point to run the simulation
├── models/                     # Trained models (.pth, .h5, .onnx) and checkpoints
├── data/                       # Datasets (small) or download scripts
│   └── raw/                    # Raw data collected from CARLA
├── notebooks/                  # Jupyter Notebooks for quick tests and analysis
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Phases

| # | Phase | Description |
|---|-------|-------------|
| 1 | **Environment Setup** | Install CARLA 0.9.15, CoppeliaSim EDU, Python stack (PyTorch, Stable Baselines 3, OpenCV, Gymnasium), and monitoring tools (TensorBoard, W&B). |
| 2 | **Modular Pipeline** | Lane detection (OpenCV), object detection (YOLOv8 fine-tuned on CARLA), Pure Pursuit planner, PID controller. Baseline metrics on 5 maps. |
| 3 | **Reinforcement Learning** | Define observation/action spaces and reward function. Train PPO and SAC agents with continuous monitoring. |
| 4 | **Imitation Learning** | Record human and autopilot demonstrations. Train BC (CNN) and CIL models after image normalization and dataset balancing. |
| 5 | **Comparative Evaluation** | Benchmark all five methods (Modular, PPO, SAC, BC, CIL) in standardized CARLA scenarios: highway, weather, intersections. |
| 6 | **CoppeliaSim Validation** | Export RL policies (SB3) and IL models (ONNX). Replicate Phase 5 evaluation protocol and measure the sim-to-sim gap. |
| 7 | **Conclusions & Docs** | Synthesize results, write final report, and prepare deliverables. |

---

## Tech Stack

- **Simulator:** CARLA 0.9.15 (UE4) · CoppeliaSim EDU (ZMQ Remote API)
- **Deep Learning:** PyTorch · Stable Baselines 3 · ONNX
- **Perception:** OpenCV · YOLOv8 (Ultralytics)
- **RL Environments:** Gymnasium
- **Monitoring:** TensorBoard · Weights & Biases
- **Language:** Python 3.10+

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/autonomous-driving-comparative.git
cd autonomous-driving-comparative

# Install dependencies
pip install -r requirements.txt
```

> **Note:** CARLA 0.9.15 and CoppeliaSim EDU must be installed separately. See `docs/administration/` for setup guides.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Route completion rate | Percentage of predefined routes completed successfully |
| Infraction-free distance | Distance traveled without collisions or traffic violations |
| Training time | Wall-clock time required to train each model |
| Policy stability | Variance in performance across repeated evaluation runs |
| OOD behavior | Performance in scenarios not seen during training |

---