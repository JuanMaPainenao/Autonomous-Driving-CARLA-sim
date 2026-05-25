"""
Script de evaluación SOLO para M3B (observación de 11 valores).

Idéntico a evaluate.py pero importa CarlaEvalEnv desde carla_eval_env_M3B
(el que genera observaciones de 11 valores compatibles con el modelo M3B).

Uso:
  python3.10 evaluate_M3B.py --town Town10HD --condition with_npcs
  python3.10 evaluate_M3B.py --all-conditions
"""

import os, csv, argparse, time
import numpy as np
from stable_baselines3 import PPO

from carla_eval_env_M3B import CarlaEvalEnv   # ← env de 11 valores
from eval_configs import (
    NUM_EPISODES, EVAL_SEEDS, TOWNS, CONDITIONS, RESULTS_DIR,
)

# Modelo M3B (definido acá para no depender del dict MODELS de eval_configs).
MODEL_NAME = "M3B"
MODEL_PATH = "models/ppo_carla_M3B_final.zip"


def compute_episode_metrics(telemetry, info, termination_cause, ep_steps):
    if not telemetry:
        return None

    speed = np.array([t['speed_kmh'] for t in telemetry])
    dist_center = np.array([t['dist_to_center'] for t in telemetry])
    orientation = np.array([t['orientation'] for t in telemetry])
    is_wrong = np.array([t['is_wrong_way'] for t in telemetry])
    steer = np.array([t['steer'] for t in telemetry])
    pos_x = np.array([t['pos_x'] for t in telemetry])
    pos_y = np.array([t['pos_y'] for t in telemetry])

    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    distance_traveled = float(np.sum(np.hypot(dx, dy)))
    steer_smoothness = float(np.var(np.diff(steer))) if len(steer) > 1 else 0.0

    return {
        'success': int(termination_cause == "success"),
        'termination_cause': termination_cause,
        'episode_steps': ep_steps,
        'survival_time_s': ep_steps * 0.05,
        'distance_traveled_m': round(distance_traveled, 2),
        'mean_speed_kmh': round(float(np.mean(speed)), 2),
        'max_speed_kmh': round(float(np.max(speed)), 2),
        'mean_dist_to_center_m': round(float(np.mean(dist_center)), 3),
        'max_dist_to_center_m': round(float(np.max(dist_center)), 3),
        'mean_orientation': round(float(np.mean(orientation)), 3),
        'wrong_way_steps': int(np.sum(is_wrong)),
        'wrong_way_pct': round(float(np.mean(is_wrong)) * 100, 2),
        'lane_invasion_count': info.get('lane_invasions', 0),
        'steering_smoothness': round(steer_smoothness, 5),
    }


def run_evaluation(town, condition_name, with_npcs):
    seed = EVAL_SEEDS[(town, with_npcs)]
    out_dir = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_{town}_{condition_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {MODEL_NAME} | {town} | {condition_name} | seed={seed}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    env = CarlaEvalEnv(town=town, with_npcs=with_npcs, seed=seed)

    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    print(f"[Eval] Cargando modelo {model_path}...")
    model = PPO.load(model_path, env=env)
    print(f"[Eval] Modelo cargado.")

    ep_csv_path = os.path.join(out_dir, "episodes.csv")
    tel_csv_path = os.path.join(out_dir, "telemetry.csv")

    ep_fields = [
        'episode', 'success', 'termination_cause', 'episode_steps', 'survival_time_s',
        'distance_traveled_m', 'mean_speed_kmh', 'max_speed_kmh',
        'mean_dist_to_center_m', 'max_dist_to_center_m', 'mean_orientation',
        'wrong_way_steps', 'wrong_way_pct', 'lane_invasion_count', 'steering_smoothness',
    ]
    tel_fields = [
        'episode', 'step', 'pos_x', 'pos_y', 'speed_kmh', 'orientation',
        'dist_to_center', 'vel_dot', 'is_wrong_way', 'action', 'steer',
        'throttle', 'brake', 'on_driving_lane', 'min_dist_npc',
    ]

    ep_file = open(ep_csv_path, 'w', newline='')
    tel_file = open(tel_csv_path, 'w', newline='')
    ep_writer = csv.DictWriter(ep_file, fieldnames=ep_fields)
    tel_writer = csv.DictWriter(tel_file, fieldnames=tel_fields)
    ep_writer.writeheader()
    tel_writer.writeheader()

    t_start = time.time()
    success_count = 0

    try:
        for ep in range(NUM_EPISODES):
            obs, _ = env.reset()
            ep_steps = 0
            termination_cause = None
            last_info = {}

            while True:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                obs, _, terminated, truncated, info = env.step(action)
                ep_steps += 1
                last_info = info
                if terminated or truncated:
                    termination_cause = info.get('termination_cause', 'unknown')
                    break

            metrics = compute_episode_metrics(env.episode_telemetry, last_info, termination_cause, ep_steps)
            if metrics is None:
                print(f"  Ep {ep+1:3d}/{NUM_EPISODES}: SIN TELEMETRÍA (saltado)")
                continue

            row = {'episode': ep + 1, **metrics}
            ep_writer.writerow(row)
            ep_file.flush()

            for t in env.episode_telemetry:
                tel_writer.writerow({'episode': ep + 1, **t})
            tel_file.flush()

            success_count += metrics['success']
            sr = success_count / (ep + 1) * 100
            elapsed = time.time() - t_start
            eta = elapsed / (ep + 1) * (NUM_EPISODES - ep - 1)
            print(f"  Ep {ep+1:3d}/{NUM_EPISODES} | {termination_cause:9s} | "
                  f"steps={ep_steps:4d} | dist={metrics['distance_traveled_m']:6.1f}m | "
                  f"v={metrics['mean_speed_kmh']:5.1f}km/h | SR={sr:5.1f}% | "
                  f"ETA={eta/60:.1f}min")

    except KeyboardInterrupt:
        print("\n=== Interrumpido ===")
    except Exception as e:
        print(f"\n=== ERROR: {type(e).__name__}: {e} ===")
        import traceback; traceback.print_exc()
    finally:
        ep_file.close()
        tel_file.close()
        env.close()
        elapsed = time.time() - t_start
        print(f"\n[Eval] Terminado en {elapsed/60:.1f} min")
        print(f"[Eval] Success rate: {success_count}/{NUM_EPISODES} ({success_count/NUM_EPISODES*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluar M3B en CARLA")
    parser.add_argument("--town", choices=TOWNS, default="Town10HD")
    parser.add_argument("--condition", choices=[c[0] for c in CONDITIONS])
    parser.add_argument("--all-conditions", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.all_conditions:
        for cn, npcs in CONDITIONS:
            run_evaluation(args.town, cn, npcs)
    else:
        if not args.condition:
            print("Especificá --condition (o usá --all-conditions)")
            return
        npcs = dict(CONDITIONS)[args.condition]
        run_evaluation(args.town, args.condition, npcs)


if __name__ == "__main__":
    main()
