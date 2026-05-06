import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env

from coppelia_env import CoppeliaEnv


# --- Hiperparámetros PPO (mismos que CARLA para coherencia) ---
PPO_HYPERPARAMS = {
    'learning_rate': 3e-4,
    'n_steps':       1024,
    'batch_size':    128,
    'n_epochs':      5,
    'gamma':         0.99,
    'gae_lambda':    0.95,
    'ent_coef':      0.01,
    'clip_range':    0.2,
    'target_kl':     0.02,
}

TOTAL_TIMESTEPS = 150_000
CHECKPOINT_FREQ = 25_000


class RewardComponentsCallback(BaseCallback):
    """Loguea los componentes individuales del reward en TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals['infos'] es una lista de dicts (uno por env paralelo)
        infos = self.locals.get('infos', [])
        for info in infos:
            for key, value in info.items():
                # Logueamos solo los componentes del reward (claves que empiezan con 'r_')
                if key.startswith('r_'):
                    self.logger.record_mean(f'reward_components/{key}', value)
                # También logueamos colisiones
                elif key == 'collision':
                    self.logger.record_mean('episode/collision_rate', float(value))
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_mode', type=str, default='R1',
                        choices=['R1', 'R2', 'R3'],
                        help='Modo de reward function')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help='Cantidad total de timesteps de entrenamiento')
    parser.add_argument('--fresh', action='store_true',
                        help='Entrena desde cero ignorando checkpoints existentes')
    args = parser.parse_args()

    reward_mode = args.reward_mode

    # --- Crear directorios de salida ---
    checkpoint_dir = f'./checkpoints/{reward_mode}'
    tb_log_dir = './tensorboard_logs'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # --- Crear entorno ---
    env = CoppeliaEnv(reward_mode=reward_mode)

    # --- Validar que el entorno cumple la API Gym ---
    print('[train] Validando entorno...')
    check_env(env)
    print('[train] Entorno OK.')

    # --- Crear o cargar modelo ---
    last_checkpoint = _find_last_checkpoint(checkpoint_dir) if not args.fresh else None

    if last_checkpoint is not None:
        print(f'[train] Cargando checkpoint: {last_checkpoint}')
        model = PPO.load(last_checkpoint, env=env, tensorboard_log=tb_log_dir)
    else:
        print(f'[train] Creando modelo nuevo (reward_mode={reward_mode})')
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=tb_log_dir,
            **PPO_HYPERPARAMS,
        )

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=checkpoint_dir,
        name_prefix=f'ppo_{reward_mode}',
    )
    components_callback = RewardComponentsCallback()

    # --- Entrenar ---
    print(f'[train] Entrenando {args.timesteps} timesteps...')
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, components_callback],
        tb_log_name=reward_mode,
        reset_num_timesteps=(last_checkpoint is None),
    )

    # --- Guardar modelo final ---
    final_path = os.path.join(checkpoint_dir, f'ppo_{reward_mode}_final')
    model.save(final_path)
    print(f'[train] Modelo final guardado en: {final_path}')

    env.close()


def _find_last_checkpoint(checkpoint_dir):
    """Busca el checkpoint más reciente en el directorio."""
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
    if not files:
        return None
    files.sort()
    return os.path.join(checkpoint_dir, files[-1])


if __name__ == '__main__':
    main()