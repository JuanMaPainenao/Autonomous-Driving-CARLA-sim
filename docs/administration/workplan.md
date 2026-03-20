# Comparative Autonomous Driving System: Modular Pipeline, Reinforcement Learning, and Imitation Learning in CARLA with Cross-Validation in CoppeliaSim

**February 2026**

---

## 1. Objectives

### 1.1 General Objectives

To develop, train, and compare three autonomous driving models (Modular Pipeline, Reinforcement Learning, and Imitation Learning) in the CARLA (Car Learning to Act) simulator, evaluating the performance of each method through standardized metrics and assessing the transferability of the trained models to a second simulation environment, CoppeliaSim, in order to draw conclusions about the limitations, practical viability, and efficiency of each model.

### 1.2 Specific Objectives

1. **Implement a classical Modular Pipeline as a baseline.** Develop an autonomous driving system based on independent perception, planning, and control modules.
2. **Train Reinforcement Learning agents in CARLA** using model-free algorithms.
3. **Implement Imitation Learning** from driving demonstrations and train Behavior Cloning and Conditional Imitation Learning models.
4. **Design a comparative evaluation methodology.** Define and measure standard metrics in the field for all three approaches: route completion rate, infraction-free distance traveled, training time, policy stability, and behavior in situations not seen during training.
5. **Validate model transferability in CoppeliaSim.** Export the models trained in CARLA and integrate them into CoppeliaSim through its Python API.
6. **Analyze the gap between simulators** and draw conclusions.

---

## 2. Activity Plan

### 2.1 Phase 1: Environment Setup

1. Installation and configuration of CARLA 0.9.15 (stable UE4 version) on a Linux/Windows environment with a compatible GPU.
2. Python environment setup: CARLA Python API, Gymnasium, Stable Baselines 3, PyTorch, OpenCV, NumPy.
3. Installation of CoppeliaSim EDU with the Python ZMQ Remote API for later use.
4. Configuration of monitoring tools: TensorBoard for training curves, Weights & Biases for experiment tracking.
5. Familiarization with the CARLA API: vehicle spawning, sensor configuration (RGB camera, virtual LiDAR, collision sensor, lane sensor).

### 2.2 Phase 2: Modular Pipeline Implementation

1. **Perception module:** lane detection with OpenCV, object detection with YOLOv8 fine-tuned on images captured from the CARLA simulator itself.
2. **Planning module:** implementation of a geometric controller (Pure Pursuit) for trajectory tracking and rule-based decision logic for traffic lights and obstacles.
3. **Control module:** PID controller for speed and steering, with explicit safety limits.
4. **Baseline metric recording:** distance traveled, number of infractions, completion rate across 5 different CARLA maps.

### 2.3 Phase 3: Reinforcement Learning in CARLA

1. Definition of the observation space: front camera image, current speed, distance to lane, traffic light indicator.
2. Definition of the continuous action space.
3. Reward function design.
4. Training with different algorithms (PPO, SAC).
5. Continuous monitoring with TensorBoard.

### 2.4 Phase 4: Imitation Learning in CARLA

1. Implementation of a teleoperation system in CARLA for recording human demonstrations using a keyboard.
2. Use of CARLA's built-in autopilot as an expert for large-scale demonstration generation.
3. Preprocessing: image normalization, dataset balancing.
4. Behavior Cloning: a CNN neural network that maps an image and a navigation command to a control action.

### 2.5 Phase 5: Comparative Evaluation in CARLA

1. Measurement of all five methods (Modular, PPO, SAC, BC, CIL) under the same conditions in CARLA using standardized metrics.
2. Evaluation in standardized scenarios (highway without traffic, weather conditions, intersections and traffic lights).
3. Construction of a global comparison table and analysis of trade-offs between methods.

### 2.6 Phase 6: Cross-Validation in CoppeliaSim

1. Setup and calibration in CoppeliaSim.
2. Transfer models as Python code. RL policies exported from Stable Baselines 3. IL models exported to ONNX format.
3. Inter-simulator gap analysis. Evaluation of each method using the unified protocol from Phase 5 in CoppeliaSim.

### 2.7 Phase 7: Conclusions and Documentation

1. Synthesis of results.
2. Conclusions.
3. Documentation and final deliverables.