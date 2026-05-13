"""
Script principal de evaluación.

Para un (modelo, town, condición) dado, corre NUM_EPISODES episodios
con deterministic=True y guarda dos CSVs:
  1. episodes.csv     → una fila por episodio (métricas agregadas)
  2. telemetry.csv    → una fila por step (datos crudos para análisis)

Uso:
  python3.10 evaluate.py --model M1 --town Town10HD --condition with_npcs
  python3.10 evaluate.py --all                 # corre las 12 combinaciones
  python3.10 evaluate.py --model M3 --all-conditions  # M3 en todo
"""

import os, csv, argparse, time, math
import numpy as np
from stable_baselines3 import PPO

from carla_eval_env import CarlaEvalEnv
from eval_configs import (
    NUM_EPISODES, MAX_STEPS, EVAL_SEEDS, MODELS, TOWNS,
    CONDITIONS, RESULTS_DIR,
)


def compute_episode_metrics(telemetry, info, termination_cause, ep_steps):
    """
    A partir de la telemetría por step, calcula métricas agregadas del episodio.
    Estas son las columnas que van a la tabla del paper.
    """
    if not telemetry:
        return None

    # numpy arrays para cálculos vectoriales rápidos.
    speed = np.array([t['speed_kmh'] for t in telemetry])
    dist_center = np.array([t['dist_to_center'] for t in telemetry])
    orientation = np.array([t['orientation'] for t in telemetry])
    is_wrong = np.array([t['is_wrong_way'] for t in telemetry])
    steer = np.array([t['steer'] for t in telemetry])
    pos_x = np.array([t['pos_x'] for t in telemetry])
    pos_y = np.array([t['pos_y'] for t in telemetry])

    # Distancia total: suma de distancias entre posiciones consecutivas.
    # np.diff calcula diferencias entre elementos adyacentes.
    # np.hypot(a, b) = sqrt(a² + b²), versión numéricamente estable.
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    distance_traveled = float(np.sum(np.hypot(dx, dy)))

    # Suavidad del steer: varianza de las diferencias entre steps consecutivos.
    # Un agente "humano" tiene steer suave (varianza baja). Un agente que
    # oscila mucho tiene varianza alta. Métrica útil para diferenciación cualitativa.
    steer_smoothness = float(np.var(np.diff(steer))) if len(steer) > 1 else 0.0

    return {
        'success': int(termination_cause == "success"),
        'termination_cause': termination_cause,
        'episode_steps': ep_steps,
        'survival_time_s': ep_steps * 0.05,  # FIXED_DELTA = 0.05
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


def run_evaluation(model_name, town, condition_name, with_npcs):
    """Corre NUM_EPISODES episodios para una combinación (modelo, town, cond)."""
    seed = EVAL_SEEDS[(town, with_npcs)]
    out_dir = os.path.join(RESULTS_DIR, f"{model_name}_{town}_{condition_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {model_name} | {town} | {condition_name} | seed={seed}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Crear env (esto carga el town, puede tardar bastante).
    env = CarlaEvalEnv(town=town, with_npcs=with_npcs, seed=seed)

    # Cargar modelo. PPO.load() restaura pesos, pero NO conecta el env hasta
    # que pasamos env=env. deterministic=True se pasa después en .predict().
    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        # SB3 acepta paths sin .zip, pero por las dudas chequeamos ambos.
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    print(f"[Eval] Cargando modelo {model_path}...")
    model = PPO.load(model_path, env=env)
    print(f"[Eval] Modelo cargado.")

    # Abrir CSVs de salida.
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
        'throttle', 'brake', 'on_driving_lane',
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
                # model.predict(): forward pass de la policy. deterministic=True
                # toma argmax sobre logits → comportamiento reproducible (mismo
                # estado siempre da misma acción). Sin esto, PPO samplea de la
                # distribución y la evaluación deja de ser determinista.
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                obs, _, terminated, truncated, info = env.step(action)
                ep_steps += 1
                last_info = info

                if terminated or truncated:
                    termination_cause = info.get('termination_cause', 'unknown')
                    break

            # Métricas del episodio.
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
        print(f"[Eval] Resultados: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelos PPO en CARLA")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Modelo a evaluar")
    parser.add_argument("--town", choices=TOWNS, help="Town")
    parser.add_argument("--condition", choices=[c[0] for c in CONDITIONS], help="Condición")
    parser.add_argument("--all", action="store_true", help="Correr las 12 combinaciones")
    parser.add_argument("--all-conditions", action="store_true", help="Para un modelo, correr las 4 condiciones")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.all:
        for m in MODELS:
            for t in TOWNS:
                for cn, npcs in CONDITIONS:
                    run_evaluation(m, t, cn, npcs)
    elif args.all_conditions:
        if not args.model:
            print("--all-conditions requiere --model")
            return
        for t in TOWNS:
            for cn, npcs in CONDITIONS:
                run_evaluation(args.model, t, cn, npcs)
    else:
        if not (args.model and args.town and args.condition):
            print("Especificá --model, --town y --condition (o usá --all / --all-conditions)")
            return
        npcs = dict(CONDITIONS)[args.condition]
        run_evaluation(args.model, args.town, args.condition, npcs)


if __name__ == "__main__":
    main()