"""
Comparison of the 5 reward designs for the paper.

Generates clean, publication-ready plots and LaTeX tables comparing
M1, M2, M3 (baselines) and M3'_A, M3'_B (refinements / ablation study).

Two views:
  - "all5": every plot includes the 5 models.
  - "ablation": extra plots focused on M3 -> M3'_A / M3'_B.

Usage:
    python3.10 compare_models.py            # generate everything
    python3.10 compare_models.py --plot sr  # only the success-rate plot
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIGURATION ──

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) == "eval":
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
else:
    RESULTS_DIR = os.path.join(BASE_DIR, "eval", "results")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All five models, in narrative order.
ALL_MODELS = ['M1', 'M2', 'M3', 'M3A', 'M3B']

# Ablation subset: original M3 vs its two refinements.
ABLATION_MODELS = ['M3', 'M3A', 'M3B']

# Display labels (what shows up on plots). M3A/M3B rendered as M3'_A / M3'_B.
LABELS = {
    'M1':  'M1',
    'M2':  'M2',
    'M3':  'M3',
    'M3A': "M3'_A",
    'M3B': "M3'_B",
}

CONDITIONS = {
    'with_npcs': 'With NPCs',
    'no_npcs':   'No NPCs',
}

TOWN = 'Town10HD'
TARGET_SPEED = 30   # km/h, training target

# Fixed color palette.
COLORS = {
    'M1':  '#E74C3C',   # red
    'M2':  '#3498DB',   # blue
    'M3':  '#2ECC71',   # green
    'M3A': '#F39C12',   # orange
    'M3B': '#9B59B6',   # purple
}


def load_episodes(model, condition):
    """Load the episodes CSV for a (model, condition). Returns None if missing."""
    path = os.path.join(RESULTS_DIR, f"{model}_{TOWN}_{condition}", "episodes.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_all():
    """Load all available DataFrames into dict {(model, cond): df}."""
    data = {}
    for m in ALL_MODELS:
        for c in CONDITIONS:
            df = load_episodes(m, c)
            if df is not None:
                data[(m, c)] = df
    return data


# ── PLOTS: 5-MODEL VIEW ──

def plot_success_rate(data, models, fname, title):
    """Grouped bar plot of Success Rate per model and condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.38

    for i, (cond_key, cond_label) in enumerate(CONDITIONS.items()):
        sr_values = []
        for m in models:
            df = data.get((m, cond_key))
            sr = df['success'].mean() * 100 if df is not None else 0
            sr_values.append(sr)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, sr_values, width,
                      label=cond_label,
                      color=[COLORS[m] for m in models],
                      alpha=0.85 if i == 0 else 0.5,
                      edgecolor='black', linewidth=0.6)
        for bar, val in zip(bars, sr_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(title)
    ax.set_ylim(0, max(15, ax.get_ylim()[1]))
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


def plot_termination(data, models, fname, title):
    """Stacked bars of termination causes. Shows each model's pathology."""
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(13, 5), sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]

    causes = ['success', 'collision', 'stall', 'offroad']
    cause_colors = {'success': '#2ECC71', 'collision': '#E74C3C',
                    'stall': '#F39C12', 'offroad': '#95A5A6'}
    cause_labels = {'success': 'Success', 'collision': 'Collision',
                    'stall': 'Stall', 'offroad': 'Off-road'}

    for ax, (cond_key, cond_label) in zip(axes, CONDITIONS.items()):
        matrix = []
        for m in models:
            df = data.get((m, cond_key))
            if df is None:
                matrix.append([0]*len(causes))
                continue
            counts = df['termination_cause'].value_counts()
            matrix.append([counts.get(c, 0) for c in causes])
        matrix = np.array(matrix)

        bottom = np.zeros(len(models))
        for j, cause in enumerate(causes):
            ax.bar([LABELS[m] for m in models], matrix[:, j], bottom=bottom,
                   label=cause_labels[cause], color=cause_colors[cause],
                   edgecolor='black', linewidth=0.5)
            bottom += matrix[:, j]

        ax.set_title(cond_label)
        ax.set_ylabel('Episodes')
        ax.set_ylim(0, 55)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


def plot_box_metrics(data, models, fname, title):
    """Box plots of 4 key continuous metrics (with_npcs condition)."""
    metrics = [
        ('distance_traveled_m', 'Distance Traveled (m)'),
        ('mean_speed_kmh', 'Mean Speed (km/h)'),
        ('mean_dist_to_center_m', 'Mean Distance to Lane Center (m)'),
        ('wrong_way_pct', 'Wrong-way Driving (%)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        cond_key = 'with_npcs'
        values, labels, box_colors = [], [], []
        for m in models:
            df = data.get((m, cond_key))
            if df is None:
                continue
            values.append(df[metric_key].values)
            labels.append(LABELS[m])
            box_colors.append(COLORS[m])

        bp = ax.boxplot(values, labels=labels, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        # Reference line at target speed for the speed subplot.
        if metric_key == 'mean_speed_kmh':
            ax.axhline(TARGET_SPEED, color='black', linestyle='--', alpha=0.5)
            ax.text(0.5, TARGET_SPEED + 1, f'Target ({TARGET_SPEED} km/h)', fontsize=8)

        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


# ── PLOTS: ANALYSIS / GAMING ──

def plot_speed_vs_distance(data, models, fname, title):
    """
    Scatter of mean_speed vs distance_traveled per episode.
    Tells the gaming story visually:
      - Models clustered at high speed + low distance => crash quickly (gaming).
      - Points reaching high distance => actually drive (M3'_A stands out).
    Each point is one episode; marker shape encodes success.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    cond_key = 'with_npcs'

    for m in models:
        df = data.get((m, cond_key))
        if df is None:
            continue
        # Successful episodes as stars, failures as circles.
        success = df[df['success'] == 1]
        fail = df[df['success'] == 0]
        ax.scatter(fail['mean_speed_kmh'], fail['distance_traveled_m'],
                   color=COLORS[m], alpha=0.5, s=30, label=f'{LABELS[m]}')
        if len(success) > 0:
            ax.scatter(success['mean_speed_kmh'], success['distance_traveled_m'],
                       color=COLORS[m], alpha=1.0, s=140, marker='*',
                       edgecolor='black', linewidth=0.8)

    ax.axvline(TARGET_SPEED, color='black', linestyle='--', alpha=0.4)
    ax.text(TARGET_SPEED + 0.5, ax.get_ylim()[1]*0.95,
            f'Target speed ({TARGET_SPEED} km/h)', fontsize=8, rotation=90, va='top')

    ax.set_xlabel('Mean Speed (km/h)')
    ax.set_ylabel('Distance Traveled (m)')
    ax.set_title(title + '\n(stars = successful episodes)')
    ax.legend(title='Model', loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


def plot_speed_distribution(data, models, fname, title):
    """
    Overlaid histograms of mean speed per episode.
    Highlights that most models overshoot the 30 km/h target (only M2 undershoots).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    cond_key = 'with_npcs'

    for m in models:
        df = data.get((m, cond_key))
        if df is None:
            continue
        ax.hist(df['mean_speed_kmh'], bins=15, alpha=0.45,
                label=LABELS[m], color=COLORS[m], edgecolor='black', linewidth=0.4)

    ax.axvline(TARGET_SPEED, color='black', linestyle='--', alpha=0.6, label=f'Target ({TARGET_SPEED} km/h)')
    ax.set_xlabel('Mean Speed (km/h)')
    ax.set_ylabel('Number of Episodes')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


# ── TABLE ──

def export_table(data):
    """Summary table (mean ± std) for all models, both conditions. CSV + LaTeX."""
    rows = []
    for m in ALL_MODELS:
        for cond_key, cond_label in CONDITIONS.items():
            df = data.get((m, cond_key))
            if df is None:
                continue
            rows.append({
                'Model': LABELS[m],
                'Condition': cond_label,
                'SR (%)': f"{df['success'].mean()*100:.1f}",
                'Distance (m)': f"{df['distance_traveled_m'].mean():.1f} ± {df['distance_traveled_m'].std():.1f}",
                'Speed (km/h)': f"{df['mean_speed_kmh'].mean():.1f} ± {df['mean_speed_kmh'].std():.1f}",
                'Lane Center (m)': f"{df['mean_dist_to_center_m'].mean():.2f} ± {df['mean_dist_to_center_m'].std():.2f}",
                'Wrong-way (%)': f"{df['wrong_way_pct'].mean():.2f}",
                'Collision (%)': f"{(df['termination_cause']=='collision').mean()*100:.0f}",
                'Stall (%)': f"{(df['termination_cause']=='stall').mean()*100:.0f}",
            })

    df_sum = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, 'comparison.csv')
    df_sum.to_csv(csv_path, index=False)
    print(f"[csv]  {csv_path}")

    tex_path = os.path.join(OUTPUT_DIR, 'comparison.tex')
    with open(tex_path, 'w') as f:
        f.write("% Auto-generated comparison table\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Comparison of the five reward designs: mean $\\pm$ std over 50 episodes per configuration}\n")
        f.write("\\label{tab:comparison}\n")
        f.write(df_sum.to_latex(index=False, escape=False,
                                column_format='l' + 'r'*(len(df_sum.columns)-1)))
        f.write("\\end{table*}\n")
    print(f"[tex]  {tex_path}")

    print("\n" + "="*90)
    print(df_sum.to_string(index=False))
    print("="*90)


# ── MAIN ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',
                        choices=['sr', 'termination', 'box', 'scatter', 'hist', 'table', 'all'],
                        default='all')
    args = parser.parse_args()

    print(f"Results dir: {RESULTS_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    data = load_all()
    print(f"Loaded {len(data)} configurations: {list(data.keys())}\n")

    if args.plot in ('sr', 'all'):
        # 5-model view
        plot_success_rate(data, ALL_MODELS, 'all5_success_rate.png',
                          f'Success Rate — All Models ({TOWN})')
        # ablation view
        plot_success_rate(data, ABLATION_MODELS, 'ablation_success_rate.png',
                          f'Success Rate — Ablation: M3 vs Refinements ({TOWN})')

    if args.plot in ('termination', 'all'):
        plot_termination(data, ALL_MODELS, 'all5_termination.png',
                         'Termination Causes — All Models')
        plot_termination(data, ABLATION_MODELS, 'ablation_termination.png',
                         'Termination Causes — Ablation: M3 vs Refinements')

    if args.plot in ('box', 'all'):
        plot_box_metrics(data, ALL_MODELS, 'all5_box_metrics.png',
                         f'Metric Distributions — All Models ({TOWN}, with NPCs)')

    if args.plot in ('scatter', 'all'):
        plot_speed_vs_distance(data, ALL_MODELS, 'all5_speed_vs_distance.png',
                               f'Speed vs Distance per Episode ({TOWN}, with NPCs)')

    if args.plot in ('hist', 'all'):
        plot_speed_distribution(data, ALL_MODELS, 'all5_speed_hist.png',
                                f'Mean Speed Distribution ({TOWN}, with NPCs)')

    if args.plot in ('table', 'all'):
        export_table(data)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()