"""
Entrenamiento de PPO + CnnPolicy en CARLA.

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
    pattern = os.path.join(checkpoint_dir, "rl_model_*_steps.zip")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    def extract_steps(path):
        # "checkpoints/rl_model_50000_steps.zip" → 50000
        basename = os.path.basename(path)
        parts = basename.replace(".zip", "").split("_")
        return int(parts[2])

    latest = max(checkpoints, key=extract_steps)
    steps_done = extract_steps(latest)
    return latest, steps_done


def create_model(env):
    """
      - CnnPolicy: red convolucional que recibe imágenes y las procesa
        con capas Conv2D antes de las capas de la política/valor.
      - n_steps=2048: cantidad de steps a recolectar antes de cada update.
        PPO acumula experiencia en un buffer y luego hace varias pasadas
        (n_epochs) sobre esa experiencia para actualizar la red.
      - batch_size=64: tamaño de los mini-batches dentro de cada epoch.
      - n_epochs=10: cuántas veces se recorre el buffer completo por update.
      - gamma=0.99: factor de descuento. Valores cercanos a 1 hacen que el
        agente valore más las recompensas futuras.
      - clip_range=0.2: rango de clipping de PPO. Limita cuánto puede cambiar
        la política en un solo update (estabilidad del entrenamiento).
      - ent_coef=0.01: coeficiente de entropía. Incentiva exploración al
        penalizar políticas muy deterministas.
      - target_kl=0.2: si la divergencia KL entre la política vieja y la
        nueva supera este umbral, se detiene el update early. Previene
        cambios demasiado drásticos en la política.
    """
    return PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,          # Era 2048 → ahora 512 para rollouts más cortos
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

    env = DummyVecEnv([lambda: Monitor(CarlaEnv(show_preview=args.preview))])

    model = create_model(env)

    if not args.fresh:
        result = find_latest_checkpoint(CHECKPOINT_DIR)
        if result is not None:
            checkpoint_path, steps_done = result
            print(f"=== Checkpoint encontrado: {checkpoint_path} ===")
            print(f"=== Steps previos: {steps_done} ===")
            loaded = PPO.load(checkpoint_path)
            model.set_parameters(loaded.get_parameters(), exact_match=True)
            del loaded  # Liberar memoria del modelo temporal

            print("=== Pesos cargados exitosamente ===")
            print("=== El contador de steps se resetea a 0 ===")

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    try:
        print(f"\n{'='*60}")
        print(f"  Iniciando entrenamiento PPO")
        print(f"  Total: {TOTAL_TIMESTEPS} steps")
        print(f"  n_steps: {model.n_steps} | batch_size: {model.batch_size}")
        print(f"  target_kl: {model.target_kl}")
        print(f"  TensorBoard: tensorboard --logdir={TENSORBOARD_DIR}")
        print(f"  Checkpoints en: {CHECKPOINT_DIR}")
        print(f"{'='*60}\n")

        print("=== Testeando env con 5 steps... ===")
        obs = env.reset()
        for i in range(5):
            obs, reward, done, info = env.step([1])  # Acción 1 = recto + throttle
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
        # Guardar modelo final pase lo que pase
        model.save(FINAL_MODEL_PATH)
        print(f"Modelo final guardado en: {FINAL_MODEL_PATH}")

        try:
            env.close()
            print("Entorno cerrado correctamente.")
        except Exception as e:
            print(f"Advertencia al cerrar entorno: {e}")

        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    main()