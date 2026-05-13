"""
Análisis de resultados de evaluación. Genera:
  - Tabla resumen con media ± std por (modelo, town, condición).
  - Tabla LaTeX lista para pegar en el paper.
  - Plots comparativos: trayectorias, box plots, bar plots.
  - Análisis de hipótesis específico para validar M3:
      * Compensation Index = corr(speed, |orientation_error|)
      * Off-axis driving rate
  - Test de significancia estadística entre modelos (Mann-Whitney U).

Uso:
  python3.10 analyze_results.py
"""

import os, csv, json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eval_configs import MODELS, TOWNS, CONDITIONS, RESULTS_DIR


def load_episodes(model, town, condition_name):
    """Carga el CSV de episodios. Devuelve DataFrame o None si no existe."""
    path = os.path.join(RESULTS_DIR, f"{model}_{town}_{condition_name}", "episodes.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_telemetry(model, town, condition_name):
    """Carga el CSV de telemetría por step."""
    path = os.path.join(RESULTS_DIR, f"{model}_{town}_{condition_name}", "telemetry.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_summary_table():
    """
    Construye la tabla resumen: para cada (modelo, town, cond), media ± std
    de las métricas principales.
    """
    rows = []
    for model in MODELS:
        for town in TOWNS:
            for cn, _ in CONDITIONS:
                df = load_episodes(model, town, cn)
                if df is None or df.empty:
                    continue
                rows.append({
                    'Modelo': model,
                    'Town': town,
                    'Condición': cn,
                    'N': len(df),
                    'SR (%)': f"{df['success'].mean()*100:.1f}",
                    'Dist (m)': f"{df['distance_traveled_m'].mean():.1f} ± {df['distance_traveled_m'].std():.1f}",
                    'Vel (km/h)': f"{df['mean_speed_kmh'].mean():.1f} ± {df['mean_speed_kmh'].std():.1f}",
                    'Centro (m)': f"{df['mean_dist_to_center_m'].mean():.2f} ± {df['mean_dist_to_center_m'].std():.2f}",
                    'Contramano (%)': f"{df['wrong_way_pct'].mean():.1f}",
                    'Invasiones': f"{df['lane_invasion_count'].mean():.1f}",
                    'Suavidad×10⁻³': f"{df['steering_smoothness'].mean()*1000:.2f}",
                })
    return pd.DataFrame(rows)


def export_latex_table(df, out_path):
    """
    Exporta la tabla a LaTeX. Usa pandas.to_latex que respeta booktabs
    (el estándar para tablas académicas).
    """
    # to_latex: convierte el DataFrame a sintaxis LaTeX. index=False evita la
    # columna de índice. escape=False permite que los símbolos como ± no se
    # escapen como \pm (los queremos literales).
    latex = df.to_latex(index=False, escape=False, column_format='l' + 'r'*(len(df.columns)-1))
    with open(out_path, 'w') as f:
        f.write("% Tabla generada automáticamente — pegar en el paper\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Resultados de evaluación — 50 episodios por configuración}\n")
        f.write("\\label{tab:eval_results}\n")
        f.write(latex)
        f.write("\\end{table*}\n")
    print(f"[LaTeX] Tabla guardada: {out_path}")


def plot_success_rate_comparison(out_dir):
    """Bar plot comparativo de Success Rate por modelo y condición."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    width = 0.25
    x_labels = [c[0] for c in CONDITIONS]

    for i, town in enumerate(TOWNS):
        ax = axes[i]
        x = np.arange(len(x_labels))
        for j, model in enumerate(MODELS):
            srs = []
            for cn, _ in CONDITIONS:
                df = load_episodes(model, town, cn)
                srs.append(df['success'].mean() * 100 if df is not None else 0)
            ax.bar(x + j * width, srs, width, label=model)
        ax.set_xticks(x + width)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{town}")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Success Rate por Modelo, Town y Condición", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "success_rate_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] {out_path}")


def plot_trajectories(town, condition_name, out_dir):
    """
    Trayectorias 2D superpuestas para los 3 modelos en una misma config.
    Una figura con 3 subplots, color verde=éxito, rojo=fallo.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for i, model in enumerate(MODELS):
        ax = axes[i]
        ep_df = load_episodes(model, town, condition_name)
        tel_df = load_telemetry(model, town, condition_name)
        if ep_df is None or tel_df is None:
            ax.set_title(f"{model} (sin datos)")
            continue

        for ep_id in ep_df['episode']:
            ep_telemetry = tel_df[tel_df['episode'] == ep_id]
            success = ep_df[ep_df['episode'] == ep_id]['success'].iloc[0]
            color = 'green' if success else 'red'
            alpha = 0.4 if success else 0.6
            ax.plot(ep_telemetry['pos_x'], ep_telemetry['pos_y'],
                    color=color, alpha=alpha, linewidth=0.8)

        sr = ep_df['success'].mean() * 100
        ax.set_title(f"{model} — SR={sr:.1f}%")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

    fig.suptitle(f"Trayectorias — {town} / {condition_name} (verde=éxito, rojo=fallo)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"trajectories_{town}_{condition_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] {out_path}")


def hypothesis_analysis(out_dir):
    """
    Análisis específico para validar la hipótesis del paper:
    "M3 (multiplicativa jerárquica) evita la 'compensación' que permiten M1/M2".

    Métricas:
      - Compensation Index: correlación entre speed y |orientation_error|.
        Positivo = el agente compensa desalineación con velocidad. M3 debería
        tener el menor (idealmente cercano a 0 o negativo).
      - Off-axis driving rate: % de steps con dist_to_center > 1.0m AND speed > 15.
        M3 debería ser el más bajo.
    """
    print("\n" + "="*60)
    print("  ANÁLISIS DE HIPÓTESIS")
    print("="*60)

    results = []
    for model in MODELS:
        for town in TOWNS:
            for cn, _ in CONDITIONS:
                tel = load_telemetry(model, town, cn)
                if tel is None or tel.empty:
                    continue

                # |orientation_error| = 1 - orientation. orientation ∈ [-1,1],
                # vale 1 si está perfectamente alineado. Su complemento es el
                # error de orientación.
                orient_error = 1.0 - tel['orientation'].values
                speed = tel['speed_kmh'].values

                # scipy.stats.pearsonr: correlación lineal de Pearson entre
                # dos arrays. Devuelve (r, p_value). r ∈ [-1, 1].
                if len(speed) > 10 and np.std(orient_error) > 1e-6:
                    comp_idx, _ = stats.pearsonr(speed, orient_error)
                else:
                    comp_idx = 0.0

                # Off-axis driving: pasos descentrados Y rápidos.
                off_axis = ((tel['dist_to_center'] > 1.0) & (tel['speed_kmh'] > 15.0))
                off_axis_pct = float(off_axis.mean()) * 100

                results.append({
                    'Modelo': model, 'Town': town, 'Condición': cn,
                    'Compensation Index': round(comp_idx, 3),
                    'Off-axis driving (%)': round(off_axis_pct, 2),
                })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, "hypothesis_analysis.csv"), index=False)

    # Plot del Compensation Index — ESTE es el plot estrella del paper si M3
    # tiene el valor más bajo en todas las configuraciones.
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot_table(index=['Town', 'Condición'], columns='Modelo',
                           values='Compensation Index')
    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Compensation Index (corr(speed, |orient_error|))")
    ax.set_title("Compensation Index por Modelo\n(más bajo = menos compensación = hipótesis validada)")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "compensation_index.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] {out_path}")


def significance_tests(out_dir):
    """
    Tests estadísticos pairwise entre modelos para Success Rate y otras métricas.

    Mann-Whitney U (scipy.stats.mannwhitneyu): test no paramétrico que compara
    dos distribuciones sin asumir normalidad. Devuelve (statistic, p_value).
    Más robusto que t-test cuando las distribuciones no son gaussianas
    (típico en Success Rate, que es binario por episodio).
    """
    print("\n" + "="*60)
    print("  TESTS DE SIGNIFICANCIA")
    print("="*60)

    rows = []
    pairs = [('M1', 'M2'), ('M1', 'M3'), ('M2', 'M3')]

    for town in TOWNS:
        for cn, _ in CONDITIONS:
            for a, b in pairs:
                df_a = load_episodes(a, town, cn)
                df_b = load_episodes(b, town, cn)
                if df_a is None or df_b is None:
                    continue
                # Test sobre distance_traveled_m (continuo, más informativo que
                # success rate binario).
                u_stat, p_val = stats.mannwhitneyu(
                    df_a['distance_traveled_m'], df_b['distance_traveled_m'],
                    alternative='two-sided'
                )
                rows.append({
                    'Town': town, 'Condición': cn, 'Comparación': f"{a} vs {b}",
                    'Métrica': 'distance_m',
                    'p-value': f"{p_val:.4f}",
                    'Significativo (p<0.05)': '✓' if p_val < 0.05 else '✗',
                })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, "significance_tests.csv"), index=False)


def main():
    analysis_dir = os.path.join(RESULTS_DIR, "_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. Tabla resumen
    print("\n" + "="*60)
    print("  TABLA RESUMEN")
    print("="*60)
    summary = compute_summary_table()
    if summary.empty:
        print("No hay resultados para analizar. Corré evaluate.py primero.")
        return
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(analysis_dir, "summary.csv"), index=False)
    export_latex_table(summary, os.path.join(analysis_dir, "summary_table.tex"))

    # 2. Plots comparativos
    plot_success_rate_comparison(analysis_dir)
    for town in TOWNS:
        for cn, _ in CONDITIONS:
            plot_trajectories(town, cn, analysis_dir)

    # 3. Análisis de hipótesis (lo más importante para el paper)
    hypothesis_analysis(analysis_dir)

    # 4. Tests de significancia
    significance_tests(analysis_dir)

    print(f"\n[Done] Todo guardado en: {analysis_dir}")


if __name__ == "__main__":
    main()