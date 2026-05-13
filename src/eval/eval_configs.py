"""
Configuración de evaluación para los 3 modelos.
Centraliza spawn points fijos, semillas y parámetros para asegurar
reproducibilidad y comparación justa entre M1, M2, M3.

Cada (town, condition) tiene su semilla fija → spawn points y orden de NPCs
son IDÉNTICOS entre los 3 modelos. La única variable es el modelo evaluado.
"""

# Episodios por configuración. 50 es estándar académico (CaRL/Roach/TransFuser
# usan entre 25 y 100). Da IC 95% suficientemente angosto con tests no paramétricos.
NUM_EPISODES = 50

# Pasos máximos por episodio. Igual al entrenamiento (2000 * 0.05s = 100s reales).
MAX_STEPS = 2000

# Step físico de la simulación (igual al entrenamiento, no se cambia).
FIXED_DELTA = 0.05

# Velocidad target en km/h (igual al entrenamiento).
TARGET_SPEED_KMH = 30

# Semillas FIJAS por (town, npcs). El mismo seed produce el mismo orden de
# spawn_points y NPCs entre M1/M2/M3 → ataque limpio a la pregunta:
# "dado el MISMO escenario, ¿qué modelo se desempeña mejor?"
EVAL_SEEDS = {
    ("Town10HD", True):  42,
    ("Town10HD", False): 43,
    ("Town02",   True):  44,
    ("Town02",   False): 45,
}

# Cantidad de NPCs cuando la condición incluye tráfico.
NUM_NPC_VEHICLES = 20

# Lista de NPCs (idéntica al training env).
NPC_VEHICLE_MODELS = [
    'dodge', 'audi', 'mini', 'mustang', 'lincoln',
    'prius', 'nissan', 'crown', 'impala',
]

# Towns a evaluar. Town10HD = entrenamiento. Town02 = generalización cross-town
# (suburbano, contraste con Town10HD urbano denso).
TOWNS = ["Town10HD", "Town02"]

# Condiciones: con NPCs (realista) y sin NPCs (ablation, aísla habilidad pura).
CONDITIONS = [
    ("with_npcs", True),
    ("no_npcs",   False),
]

# Modelos a evaluar. La key es el nombre corto, el value es el path al .zip.
MODELS = {
    "M1": "models/ppo_carla_M1_final.zip",
    "M2": "models/ppo_carla_M2_final.zip",
    "M3": "models/ppo_carla_M3_final.zip",
}

# Directorio raíz para resultados.
RESULTS_DIR = "eval/results"

# Timeout del cliente en segundos. Alto porque load_world puede tardar 60-90s
# al cambiar de mapa, sobre todo Town10HD ↔ Town05/Town02 en GPUs con poca VRAM.
CLIENT_TIMEOUT = 120.0

# Parámetros del vehículo / sensores (idénticos al training env).
IM_WIDTH, IM_HEIGHT, IM_CHANNELS, FOV = 640, 480, 3, 110
OBS_WIDTH, OBS_HEIGHT = 160, 120
VECTOR_OBS_SIZE = 10
STALL_TERMINATE_STEPS = 30

# Acciones discretas (idénticas al training env, NO se tocan).
DISCRETE_ACTIONS = {
    0: (-0.3, 0.6, 0.0),   1: ( 0.0, 0.6, 0.0),   2: ( 0.3, 0.6, 0.0),
    3: (-0.3, 1.0, 0.0),   4: ( 0.0, 1.0, 0.0),   5: ( 0.3, 1.0, 0.0),
    6: (-0.3, 0.0, 0.5),   7: ( 0.0, 0.0, 0.5),   8: ( 0.3, 0.0, 0.5),
}