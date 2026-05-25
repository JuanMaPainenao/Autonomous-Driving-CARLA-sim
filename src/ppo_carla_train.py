"""
Entrenamiento PPO + MultiInputPolicy — Modelo 3'_A.
Hiperparámetros IDÉNTICOS a M1/M2/M3 (comparación justa).

Uso:
    python3.10 ppo_carla_train_M3A.py                # entrenar
    python3.10 ppo_carla_train_M3A.py --preview      # con ventana
    python3.10 ppo_carla_train_M3A.py --fresh        # ignorar checkpoints
"""

import os, glob, argparse, signal
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from carla_env_M3A import CarlaEnv

TOTAL_TIMESTEPS = 600_000
CHECKPOINT_FREQ = 20_000

CHECKPOINT_DIR = "checkpoints_M3A/"
TENSORBOARD_DIR = "tensorboard/"
FINAL_MODEL_PATH = "models/ppo_carla_M3A_final"


def find_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "rl_model_*_steps.zip")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    def extract_steps(p):
        return int(os.path.basename(p).replace(".zip","").split("_")[2])
    latest = max(checkpoints, key=extract_steps)
    return latest, extract_steps(latest)


def create_model(env):
    return PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        target_kl=0.02,
        verbose=1,
        tensorboard_log=TENSORBOARD_DIR,
        device="auto",
    )


def main():
    parser = argparse.ArgumentParser(description="Entrenar PPO — Modelo 3'_A")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(CarlaEnv(show_preview=args.preview))])
    resumed = False

    if not args.fresh:
        result = find_latest_checkpoint(CHECKPOINT_DIR)
        if result is not None:
            path, steps = result
            print(f"=== Checkpoint: {path} ({steps} steps) ===")
            try:
                model = PPO.load(path, env=env, tensorboard_log=TENSORBOARD_DIR)
                resumed = True
                print(f"=== Modelo cargado, num_timesteps={model.num_timesteps} ===")
            except Exception as e:
                print(f"=== Error al cargar: {e}. Usá --fresh ===")
                try: env.close()
                except: pass
                os.kill(os.getpid(), signal.SIGTERM)

    if not resumed:
        model = create_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="rl_model", save_replay_buffer=False, save_vecnormalize=False,
    )

    try:
        print(f"\n{'='*50}")
        print(f"  Modelo 3'_A: Multiplicativo + Collision Penalty x5")
        print(f"  Steps: {TOTAL_TIMESTEPS} | n_steps: 1024 | batch: 128")
        print(f"  epochs: 5 | lr: 3e-4 | ent_coef: 0.01")
        print(f"{'='*50}\n")

        obs = env.reset()
        for i in range(5):
            obs, reward, done, info = env.step([1])
            print(f"  Test step {i}: reward={reward[0]:.2f}")
        print("=== Env OK, arrancando ===")

        remaining = TOTAL_TIMESTEPS - model.num_timesteps
        print(f"  Steps restantes: {remaining}")

        model.learn(
            total_timesteps=remaining,
            callback=checkpoint_cb,
            reset_num_timesteps=False,
            tb_log_name="M3A_collision_x5",
            progress_bar=True,
        )

    except KeyboardInterrupt:
        print("\n=== Interrumpido ===")
    except Exception as e:
        print(f"\n=== ERROR: {type(e).__name__}: {e} ===")
        import traceback; traceback.print_exc()
    finally:
        model.save(FINAL_MODEL_PATH)
        print(f"Modelo guardado: {FINAL_MODEL_PATH}")
        try: env.close()
        except: pass
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    main()