"""
Entrenamiento de PPO + MultiInputPolicy en CARLA.

Uso:
    1. Iniciar CARLA:
       cd ~/Downloads/CARLA/CARLA_0.9.15/
       __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./CarlaUE4.sh

    2. Entrenar (sin ventana, más rápido):
       python3.10 ppo_carla_train.py

    3. Entrenar con preview:
       python3.10 ppo_carla_train.py --preview

    4. Entrenar desde cero ignorando checkpoints:
       python3.10 ppo_carla_train.py --fresh

    5. Ver TensorBoard:
       tensorboard --logdir=./tensorboard/

IMPORTANTE: Esta versión usa carla_env.py v2 (wrong_way + stall progresivo).
Los checkpoints de la versión anterior (v1) NO son compatibles porque el
observation_space cambió de 7 a 9 valores en el vector. Usar --fresh para
entrenar desde cero con la nueva reward.
"""

import os
import glob
import argparse
import signal

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from carla_env import CarlaEnv

TOTAL_TIMESTEPS = 500_000     # Steps totales de entrenamiento
CHECKPOINT_FREQ = 10_000      # Guardar checkpoint cada N steps
CHECKPOINT_DIR = "checkpoints/"
TENSORBOARD_DIR = "tensorboard/"
FINAL_MODEL_PATH = "models/ppo_carla_final"


def find_latest_checkpoint(checkpoint_dir):
    """
    Busca el checkpoint más reciente en el directorio.
    glob.glob() busca archivos que coincidan con un patrón con wildcards.
    """
    pattern = os.path.join(checkpoint_dir, "rl_model_*_steps.zip")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    def extract_steps(path):
        basename = os.path.basename(path)
        parts = basename.replace(".zip", "").split("_")
        return int(parts[2])

    latest = max(checkpoints, key=extract_steps)
    steps_done = extract_steps(latest)
    return latest, steps_done


def create_model(env):
    """
    Crea un modelo PPO con MultiInputPolicy.

    MultiInputPolicy acepta observaciones Dict (imagen + vector).
    Internamente usa CombinedExtractor que:
      1. Procesa "image" con NatureCNN (capas convolucionales)
      2. Procesa "vector" con capas fully-connected
      3. Concatena ambas representaciones
      4. Alimenta las cabezas de política y valor
    """
    return PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.2,
        verbose=1,
        tensorboard_log=TENSORBOARD_DIR,
        device="auto",
    )


def main():
    parser = argparse.ArgumentParser(description="Entrenar PPO en CARLA")
    parser.add_argument(
        "--preview", action="store_true",
        help="Mostrar ventana con la vista del agente",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignorar checkpoints y entrenar desde cero",
    )
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # DummyVecEnv: wrapper que convierte un solo entorno en un "vectorizado"
    # de 1 entorno. SB3 requiere entornos vectorizados.
    # Monitor: wrapper que registra ep_reward y ep_length para TensorBoard.
    env = DummyVecEnv([lambda: Monitor(CarlaEnv(show_preview=args.preview))])

    model = create_model(env)

    if not args.fresh:
        result = find_latest_checkpoint(CHECKPOINT_DIR)
        if result is not None:
            checkpoint_path, steps_done = result
            print(f"=== Checkpoint encontrado: {checkpoint_path} ===")
            print(f"=== Steps previos: {steps_done} ===")

            try:
                loaded = PPO.load(checkpoint_path)
                model.set_parameters(loaded.get_parameters(), exact_match=True)
                del loaded
                print("=== Pesos cargados exitosamente ===")
            except Exception as e:
                print(f"=== ERROR al cargar checkpoint: {e} ===")
                print("=== El vector de observación cambió de 7 a 9. ===")
                print("=== Usá --fresh para entrenar desde cero. ===")
                try:
                    env.close()
                except Exception:
                    pass
                # os.kill en vez de return: evita que Python intente
                # destruir actores CARLA de nuevo al salir del proceso,
                # lo que causa "trying to operate on a destroyed actor".
                os.kill(os.getpid(), signal.SIGTERM)

            print("=== El contador de steps se resetea a 0 ===")

    # CheckpointCallback: guarda los pesos de la red cada CHECKPOINT_FREQ
    # steps a disco como archivos .zip.
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    try:
        print(f"\n{'='*60}")
        print(f"  Iniciando entrenamiento PPO + MultiInputPolicy v2")
        print(f"  (wrong_way + stall progresivo)")
        print(f"  Total: {TOTAL_TIMESTEPS} steps")
        print(f"  n_steps: {model.n_steps} | batch_size: {model.batch_size}")
        print(f"  target_kl: {model.target_kl}")
        print(f"  TensorBoard: tensorboard --logdir={TENSORBOARD_DIR}")
        print(f"  Checkpoints en: {CHECKPOINT_DIR}")
        print(f"{'='*60}\n")

        # Test rápido del env antes de entrenar
        print("=== Testeando env con 5 steps... ===")
        obs = env.reset()
        for i in range(5):
            obs, reward, done, info = env.step([1])
            print(f"  Step {i}: reward={reward[0]:.2f}, done={done[0]}")
        print("=== Env OK, arrancando entrenamiento ===")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_cb,
            reset_num_timesteps=True,
            tb_log_name="PPO_CARLA",
            progress_bar=True,
        )

    except KeyboardInterrupt:
        print("\n=== Entrenamiento interrumpido por el usuario ===")

    except Exception as e:
        print(f"\n=== ERROR durante el entrenamiento ===")
        print(f"=== Tipo: {type(e).__name__} ===")
        print(f"=== Detalle: {e} ===")
        import traceback
        traceback.print_exc()

    finally:
        model.save(FINAL_MODEL_PATH)
        print(f"Modelo final guardado en: {FINAL_MODEL_PATH}")

        try:
            env.close()
            print("Entorno cerrado correctamente.")
        except Exception as e:
            print(f"Advertencia al cerrar entorno: {e}")

        # os.kill con SIGTERM: fuerza terminación limpia del proceso.
        # Necesario porque CARLA a veces deja threads colgados.
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    main()