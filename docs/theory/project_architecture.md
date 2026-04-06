# Project Architecture

## 1. Overview

This project trains an autonomous driving agent using PPO (Proximal Policy Optimization) inside the CARLA simulator. The code is built on top of two main libraries: **Gymnasium** (the environment interface) and **Stable Baselines3** (the RL algorithm). This document explains how these pieces fit together.

The high-level flow is:

```
CARLA Simulator  ←→  CarlaEnv (Gymnasium)  ←→  PPO (Stable Baselines3)
```

CARLA provides the simulated world (physics, rendering, sensors). `CarlaEnv` wraps it into the standard Gymnasium interface. SB3's PPO reads observations from the environment, decides actions, and updates the neural network.

## 2. Gymnasium

Gymnasium (formerly OpenAI Gym) is the standard API for RL environments in Python [1]. It defines a common interface that all environments must follow, so any RL algorithm can work with any environment.

### 2.1 The `gym.Env` Interface

Every Gymnasium environment is a class that inherits from `gym.Env` and implements:

```python
class MyEnv(gym.Env):
    def __init__(self):
        self.observation_space = ...  # what the agent sees
        self.action_space = ...       # what the agent can do

    def reset(self):
        # restart episode, return (observation, info)
        ...

    def step(self, action):
        # execute action, return (observation, reward, terminated, truncated, info)
        ...

    def close(self):
        # cleanup resources
        ...
```

The two key attributes are:

- `observation_space` — defines the shape and type of observations (what the agent "sees")
- `action_space` — defines the set of valid actions (what the agent can "do")

### 2.2 The Step Loop

Every RL training loop follows the same pattern:

```python
obs, info = env.reset()

while not done:
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

- `terminated` — the episode ended naturally (e.g., collision)
- `truncated` — the episode was cut short (e.g., max steps reached)

This distinction matters for RL: if an episode was truncated, the agent knows it didn't actually fail — it just ran out of time [1].

### 2.3 Spaces

Gymnasium uses `Space` objects to describe observations and actions:

```python
from gymnasium import spaces

# Image pixels: 120×160×3, values 0-255
spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)

# 9 possible actions: 0, 1, 2, ..., 8
spaces.Discrete(9)

# Mixed observation (image + vector)
spaces.Dict({
    "image": spaces.Box(...),
    "vector": spaces.Box(low=-1.0, high=2.0, shape=(7,), dtype=np.float32),
})
```

`spaces.Dict` is what enables multi-input observations — SB3's `MultiInputPolicy` detects this and creates the appropriate network architecture automatically.

## 3. Stable Baselines3 (SB3)

SB3 is a library of reliable RL algorithm implementations in PyTorch [2]. It provides ready-to-use algorithms (PPO, SAC, DQN, A2C, etc.) that work with any Gymnasium-compatible environment.

### 3.1 Basic Usage

```python
from stable_baselines3 import PPO

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("my_model")
```

That's it. SB3 handles the entire training loop internally: collecting experience, computing advantages, updating the network, logging to TensorBoard.

### 3.2 Policy Types

SB3 picks the network architecture based on the policy string:

- `"MlpPolicy"` — fully-connected network, for vector observations
- `"CnnPolicy"` — convolutional network, for image observations
- `"MultiInputPolicy"` — combined CNN + MLP, for `Dict` observation spaces

With `MultiInputPolicy`, SB3 uses a `CombinedExtractor` that processes each key in the dict with the right sub-network and concatenates the outputs.

### 3.3 Wrappers

SB3 uses Gymnasium wrappers to add functionality without modifying the environment:

```python
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

env = DummyVecEnv([lambda: Monitor(CarlaEnv())])
```

- `Monitor` — records episode rewards and lengths for logging
- `DummyVecEnv` — wraps the environment in SB3's vectorized interface (required by SB3, even for a single environment)

### 3.4 Callbacks

Callbacks let you run custom code during training without modifying the training loop:

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_cb = CheckpointCallback(
    save_freq=10_000,         # save every N steps
    save_path="checkpoints/",
    name_prefix="rl_model",
)

model.learn(total_timesteps=500_000, callback=checkpoint_cb)
```

`CheckpointCallback` saves the model weights periodically, so training can be resumed if interrupted.

## 4. Project Structure

The project consists of two main files:

### 4.1 `carla_env.py` — The Environment

This file defines `CarlaEnv`, which inherits from `gym.Env` and bridges CARLA with Gymnasium.

```
CarlaEnv(gym.Env)
│
├── __init__()
│   ├── Define observation_space (Dict: image + vector)
│   ├── Define action_space (Discrete)
│   ├── Connect to CARLA (client, world, map)
│   ├── Set synchronous mode
│   └── Spawn NPC traffic
│
├── reset()
│   ├── Destroy previous episode actors
│   ├── Spawn ego vehicle + sensors (camera, collision, lane)
│   ├── Wait for first camera frame
│   └── Return (observation_dict, info)
│
├── step(action)
│   ├── Apply vehicle control (steer, throttle, brake)
│   ├── Tick simulation
│   ├── Compute reward
│   ├── Build observation dict
│   └── Return (obs, reward, terminated, truncated, info)
│
├── _compute_reward()
│   └── Calculate: speed + orientation + progress + lane + collision + offroad + stall
│
├── _build_obs()
│   └── Return {"image": camera_frame, "vector": measurements}
│
├── _get_vector_obs()
│   └── Build normalized vector: [speed, collision, orientation, dist_to_center, angle_diff, offroad, stall]
│
└── close()
    └── Destroy actors, restore CARLA settings
```

### 4.2 `ppo_carla_train.py` — The Training Script

This file creates the environment, configures PPO, and runs training.

```
main()
│
├── Parse arguments (--preview, --fresh)
├── Create environment: DummyVecEnv([Monitor(CarlaEnv())])
├── Create PPO model with MultiInputPolicy
│
├── Load checkpoint (if not --fresh)
│   ├── Find latest checkpoint in checkpoints/
│   └── Load weights with set_parameters()
│
├── Train
│   ├── model.learn(total_timesteps, callback)
│   └── CheckpointCallback saves every N steps
│
└── Finally
    ├── Save final model
    └── Close environment
```

## 5. CARLA Synchronous Mode

By default CARLA runs asynchronously — the simulation advances in real time regardless of the client. For RL training, this is problematic because the agent needs deterministic, reproducible steps.

Synchronous mode forces CARLA to advance exactly one fixed timestep per `world.tick()` call:

```python
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)
```

Each call to `world.tick()` advances the simulation by exactly 0.05 seconds. No more, no less. This ensures that the physics are deterministic and the agent always gets consistent observations [3].

The Traffic Manager must also be set to synchronous mode, otherwise NPC vehicles desync.

## 6. Sensors in CARLA

CARLA provides sensors that attach to actors and stream data via callbacks:

```python
# Camera: captures RGB images
camera.listen(lambda image: self._process_image(image))

# Collision: fires when the vehicle hits something
collision_sensor.listen(lambda event: self._on_collision(event))

# Lane invasion: fires when the vehicle crosses a lane marking
lane_sensor.listen(lambda event: self._on_lane_invasion(event))
```

These callbacks run asynchronously in CARLA's thread. The environment reads the latest data when `step()` is called [3].

## 7. Data Flow

Each training step follows this path:

```
PPO selects action (integer 0-8)
        │
        ▼
CarlaEnv.step(action)
        │
        ├── Translate action → (steer, throttle, brake)
        ├── vehicle.apply_control(...)
        ├── world.tick()  →  CARLA advances simulation
        │
        ├── Camera callback stores new frame
        ├── Collision/Lane sensors update flags
        │
        ├── _compute_reward()  →  reward float
        ├── _build_obs()       →  {"image": ..., "vector": ...}
        │
        └── Return (obs, reward, terminated, truncated, info)
                │
                ▼
        PPO stores transition in rollout buffer
        PPO updates network every n_steps
```

## References

[1] Gymnasium Documentation. Farama Foundation. https://gymnasium.farama.org/

[2] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable reinforcement learning implementations," *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

[3] CARLA Documentation. https://carla.readthedocs.io/
