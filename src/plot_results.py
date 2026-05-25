"""
Analysis and visualization of evaluation results for the paper.

Generates plots and comparative tables ready for inclusion in LaTeX.
Each function generates ONE type of plot — easy to modify/extend.

Usage:
    python3.10 plot_results.py             # generate all plots
    python3.10 plot_results.py --plot bar  # only the bar plot
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIGURATION ──

# __file__ is the path of this script. abspath() turns it into an absolute path
# and dirname() extracts the directory. This makes the script work regardless
# of which directory you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-detect results directory: if script lives inside eval/, results is next
# to it; otherwise (script in src/) results is inside eval/.
if os.path.basename(BASE_DIR) == "eval":
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
else:
    RESULTS_DIR = os.path.join(BASE_DIR, "eval", "results")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models to compare — add 'M3A' and 'M3B' when their evaluations are done.
MODELS = ['M1', 'M2', 'M3']

# Conditions (folder names) and their labels for plots.
CONDITIONS = {
    'with_npcs': 'With NPCs',
    'no_npcs':   'No NPCs',
}

TOWN = 'Town10HD'   # only town with complete evaluation

# Fixed color palette for consistency across plots. Always use the same colors.
COLORS = {
    'M1':  '#E74C3C',   # red (aggressive)
    'M2':  '#3498DB',   # blue (cautious)
    'M3':  '#2ECC71',   # green (original hypothesis)
    'M3A': '#F39C12',   # orange (quantitative refinement)
    'M3B': '#9B59B6',   # purple (structural refinement)
}


def load_episodes(model, condition):
    """Load the CSV for a (model, condition) pair. Returns None if missing."""
    path = os.path.join(RESULTS_DIR, f"{model}_{TOWN}_{condition}", "episodes.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_all():
    """Load all available DataFrames. Returns dict {(model, cond): df}."""
    data = {}
    for m in MODELS:
        for c in CONDITIONS:
            df = load_episodes(m, c)
            if df is not None:
                data[(m, c)] = df
    return data


# ── PLOTS ──

def plot_success_rate_bar(data):
    """
    Bar plot: Success Rate per model and condition.
    Main plot of the paper: shows which model wins.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODELS))      # bar positions: [0, 1, 2]
    width = 0.35                    # width of each bar

    # One bar series per condition (with/no NPCs), grouped by model.
    for i, (cond_key, cond_label) in enumerate(CONDITIONS.items()):
        # Compute SR for each model in this condition.
        sr_values = []
        for m in MODELS:
            df = data.get((m, cond_key))
            sr = df['success'].mean() * 100 if df is not None else 0
            sr_values.append(sr)

        # offset: horizontal shift so bars don't overlap.
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, sr_values, width,
                      label=cond_label,
                      color=[COLORS[m] for m in MODELS],
                      alpha=0.8 if i == 0 else 0.5,
                      edgecolor='black', linewidth=0.5)

        # Numeric labels above each bar.
        for bar, val in zip(bars, sr_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Success Rate by Model — {TOWN}')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'success_rate.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


def plot_termination_distribution(data):
    """
    Stacked bar plot: distribution of termination causes per model.
    Excellent for showing the DIFFERENT pathologies of each model.
    """
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(12, 5), sharey=True)

    # plt.subplots with 1 condition returns a single ax, not an array. Normalize.
    if len(CONDITIONS) == 1:
        axes = [axes]

    causes = ['success', 'collision', 'stall', 'offroad']
    cause_colors = {'success': '#2ECC71', 'collision': '#E74C3C',
                    'stall': '#F39C12', 'offroad': '#95A5A6'}
    cause_labels = {'success': 'Success', 'collision': 'Collision',
                    'stall': 'Stall', 'offroad': 'Off-road'}

    for ax, (cond_key, cond_label) in zip(axes, CONDITIONS.items()):
        # Build matrix: rows = models, columns = causes.
        matrix = []
        for m in MODELS:
            df = data.get((m, cond_key))
            if df is None:
                matrix.append([0]*len(causes))
                continue
            # value_counts counts occurrences of each unique value.
            counts = df['termination_cause'].value_counts()
            row = [counts.get(c, 0) for c in causes]   # .get(c, 0) = 0 if missing
            matrix.append(row)
        matrix = np.array(matrix)

        # Stacked bars: each bar is a model, each color is a cause.
        bottom = np.zeros(len(MODELS))
        for j, cause in enumerate(causes):
            ax.bar(MODELS, matrix[:, j], bottom=bottom,
                   label=cause_labels[cause], color=cause_colors[cause],
                   edgecolor='black', linewidth=0.5)
            bottom += matrix[:, j]

        ax.set_title(cond_label)
        ax.set_ylabel('Episodes')
        ax.set_ylim(0, 55)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.suptitle('Distribution of Termination Causes', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'termination_distribution.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


def plot_box_metrics(data):
    """
    Box plots of the most important continuous metrics.
    Shows the DISTRIBUTION, not just the mean. Reveals outliers.
    """
    metrics = [
        ('distance_traveled_m', 'Distance Traveled (m)'),
        ('mean_speed_kmh', 'Mean Speed (km/h)'),
        ('mean_dist_to_center_m', 'Mean Distance to Lane Center (m)'),
        ('wrong_way_pct', 'Wrong-way Driving (%)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()   # convert 2x2 to array of 4

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        # We focus on a single condition (with_npcs) for simplicity.
        cond_key = 'with_npcs'

        # Collect values per model in a list.
        # matplotlib's boxplot() expects a list of arrays.
        values = []
        labels = []
        for m in MODELS:
            df = data.get((m, cond_key))
            if df is None:
                continue
            values.append(df[metric_key].values)   # .values = numpy array
            labels.append(m)

        # patch_artist=True allows coloring the boxes.
        bp = ax.boxplot(values, labels=labels, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})

        # Assign colors to each box according to model.
        for patch, m in zip(bp['boxes'], labels):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.7)

        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle(f'Distribution of Metrics — {TOWN} with NPCs', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'box_metrics.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


def plot_speed_histogram(data):
    """
    Histogram of mean speeds per episode.
    Shows the 'signature' of each model: M1 high, M2 low, M3 medium-high.
    """
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(12, 4), sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]

    for ax, (cond_key, cond_label) in zip(axes, CONDITIONS.items()):
        for m in MODELS:
            df = data.get((m, cond_key))
            if df is None:
                continue
            # alpha=0.5 so overlapping histograms remain visible.
            # bins=15 = 15 bars (enough for 50 episodes).
            ax.hist(df['mean_speed_kmh'], bins=15, alpha=0.5,
                    label=m, color=COLORS[m], edgecolor='black')

        # Vertical line at TARGET_SPEED = 30 km/h, for reference.
        ax.axvline(30, color='black', linestyle='--', alpha=0.5, label='Target (30 km/h)')

        ax.set_xlabel('Mean Speed (km/h)')
        ax.set_ylabel('Number of Episodes')
        ax.set_title(cond_label)
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle('Distribution of Mean Speeds per Episode', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'speed_histogram.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[plot] {path}")


def export_summary_table(data):
    """
    Summary table with mean ± std of all important metrics.
    Exports to CSV and to LaTeX for inclusion in the paper.
    """
    rows = []
    for m in MODELS:
        for cond_key, cond_label in CONDITIONS.items():
            df = data.get((m, cond_key))
            if df is None:
                continue
            rows.append({
                'Model': m,
                'Condition': cond_label,
                'SR (%)': f"{df['success'].mean()*100:.1f}",
                'Distance (m)': f"{df['distance_traveled_m'].mean():.1f} ± {df['distance_traveled_m'].std():.1f}",
                'Speed (km/h)': f"{df['mean_speed_kmh'].mean():.1f} ± {df['mean_speed_kmh'].std():.1f}",
                'Lane Center (m)': f"{df['mean_dist_to_center_m'].mean():.2f} ± {df['mean_dist_to_center_m'].std():.2f}",
                'Wrong-way (%)': f"{df['wrong_way_pct'].mean():.2f}",
                'Lane Invasions': f"{df['lane_invasion_count'].mean():.1f}",
                'Collision (%)': f"{(df['termination_cause']=='collision').mean()*100:.0f}",
                'Stall (%)': f"{(df['termination_cause']=='stall').mean()*100:.0f}",
            })

    df_summary = pd.DataFrame(rows)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"[csv]  {csv_path}")

    # LaTeX
    tex_path = os.path.join(OUTPUT_DIR, 'summary.tex')
    with open(tex_path, 'w') as f:
        f.write("% Automatically generated table\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Evaluation metrics: mean $\\pm$ standard deviation over 50 episodes}\n")
        f.write("\\label{tab:results}\n")
        # .to_latex generates the body of tabular. escape=False keeps ± as literal.
        f.write(df_summary.to_latex(index=False, escape=False,
                                     column_format='l' + 'r'*(len(df_summary.columns)-1)))
        f.write("\\end{table*}\n")
    print(f"[tex]  {tex_path}")

    # Also print to terminal for quick inspection.
    print("\n" + "="*80)
    print(df_summary.to_string(index=False))
    print("="*80)


# ── MAIN ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', choices=['bar', 'termination', 'box', 'hist', 'table', 'all'],
                        default='all')
    args = parser.parse_args()

    print(f"Results dir: {RESULTS_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print("Loading data...")
    data = load_all()
    print(f"Loaded {len(data)} configurations: {list(data.keys())}")
    print()

    if args.plot in ('bar', 'all'):
        plot_success_rate_bar(data)
    if args.plot in ('termination', 'all'):
        plot_termination_distribution(data)
    if args.plot in ('box', 'all'):
        plot_box_metrics(data)
    if args.plot in ('hist', 'all'):
        plot_speed_histogram(data)
    if args.plot in ('table', 'all'):
        export_summary_table(data)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()