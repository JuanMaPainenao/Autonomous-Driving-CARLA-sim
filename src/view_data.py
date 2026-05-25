import pandas as pd

# Cargar los 3 CSVs en un diccionario
dfs = {
    'M1': pd.read_csv('eval/results/M1_Town10HD_with_npcs/episodes.csv'),
    'M2': pd.read_csv('eval/results/M2_Town10HD_with_npcs/episodes.csv'),
    'M3': pd.read_csv('eval/results/M3_Town10HD_with_npcs/episodes.csv'),
    'M3A': pd.read_csv('eval/results/M3A_Town10HD_with_npcs/episodes.csv'),
}

# Para cada métrica que te interese, comparar los promedios
metrics = ['distance_traveled_m', 'mean_speed_kmh', 'mean_dist_to_center_m', 'wrong_way_pct']

for metric in metrics:
    print(f"\n=== {metric} ===")
    for name, df in dfs.items():
        # df[metric] selecciona la columna. .mean() y .std() son auto-explicativos.
        m = df[metric].mean()
        s = df[metric].std()
        print(f"  {name}: {m:.2f} ± {s:.2f}")



