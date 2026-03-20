"""
Entorno Gymnasium para CARLA con función de recompensa detallada.
Diseñado para usar con PPO + CnnPolicy de Stable Baselines3.

Características:
  - Modo síncrono (deterministic ticks, esencial para RL estable)
  - Acciones discretas: 9 combinaciones de steer × throttle/brake
  - Recompensas por componentes: velocidad, orientación, lane, colisión
  - Logging de recompensas a CSV para análisis
  - Checkpoints automáticos cada N steps

Uso:
    python3.10 ppo_carla_train.py
"""

import glob
import os
import sys
import random
import time
import math
import csv
from datetime import datetime

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

# ── CARLA import ──────────────────────────────────────────────────
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════
IM_WIDTH = 160           # Ancho de imagen (más chica = más rápido)
IM_HEIGHT = 120          # Alto de imagen
IM_CHANNELS = 3          # RGB
FOV = 110                # Campo de visión de la cámara en grados

FIXED_DELTA = 0.05       # 20 ticks/segundo en modo síncrono (1/0.05 = 20 FPS simulados)
MAX_STEPS = 2000         # Máximo de steps por episodio (2000 × 0.05s = 100s simulados)

# Velocidad objetivo en km/h
TARGET_SPEED_KMH = 40

# ── Magnitudes de recompensa ─────────────────────────────────────
# Calibradas para que las recompensas positivas puedan superar a las penalidades.
# Si las penalidades son mucho mayores que las recompensas, el agente aprende
# a quedarse quieto o girar en círculos (óptimo local de "evitar castigo").
R_SPEED_MAX = 1.0        # Recompensa máxima por velocidad adecuada
R_ORIENTATION_MAX = 1.0  # Recompensa máxima por estar alineado con la ruta
R_LANE_PENALTY = -5.0    # Penalidad por invasión de carril
R_COLLISION_PENALTY = -10.0   # Penalidad por colisión
R_OFFROAD_PENALTY = -5.0      # Penalidad por salirse de la ruta

# ── NPC Traffic ──────────────────────────────────────────────────
NUM_NPC_VEHICLES = 20        # Cantidad de vehículos NPC con autopilot
# Modelos de vehículos para NPCs (variedad visual)
NPC_VEHICLE_MODELS = ['dodge', 'audi', 'mini', 'mustang', 'lincoln',
                      'prius', 'nissan', 'crown', 'impala']

# ── Ventana de preview ───────────────────────────────────────────
# Resolución de la ventana de cv2 que muestra lo que ve el agente.
# Es independiente de IM_WIDTH/IM_HEIGHT (la obs del modelo).
# La imagen se escala al mostrarla para que sea visible.
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480


# ══════════════════════════════════════════════════════════════════
#  ACCIONES DISCRETAS
# ══════════════════════════════════════════════════════════════════
# 9 combinaciones de (steer, throttle/brake).
# Cada tupla es (steer, throttle, brake).
# steer: -0.3 (izquierda), 0 (recto), 0.3 (derecha)
# Los valores de steer son moderados (no ±1.0) para evitar giros bruscos.
DISCRETE_ACTIONS = {
    0: (-0.3, 0.6, 0.0),   # Izquierda + throttle medio
    1: ( 0.0, 0.6, 0.0),   # Recto + throttle medio
    2: ( 0.3, 0.6, 0.0),   # Derecha + throttle medio
    3: (-0.3, 1.0, 0.0),   # Izquierda + throttle full
    4: ( 0.0, 1.0, 0.0),   # Recto + throttle full
    5: ( 0.3, 1.0, 0.0),   # Derecha + throttle full
    6: (-0.3, 0.0, 0.5),   # Izquierda + freno
    7: ( 0.0, 0.0, 0.5),   # Recto + freno
    8: ( 0.3, 0.0, 0.5),   # Derecha + freno
}


# ══════════════════════════════════════════════════════════════════
#  ENTORNO GYMNASIUM
# ══════════════════════════════════════════════════════════════════
class CarlaEnv(gym.Env):
    """
    Wrapper Gymnasium para CARLA.

    gymnasium.Env es la clase base que define la interfaz estándar:
      - observation_space: qué tipo de datos recibe el agente (imagen en este caso)
      - action_space: qué acciones puede tomar (9 discretas)
      - reset(): reinicia el episodio y retorna observación inicial
      - step(action): ejecuta la acción, retorna (obs, reward, terminated, truncated, info)

    SB3 espera que el env siga esta interfaz para poder entrenar.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, show_preview=False):
        super().__init__()

        # ── Espacios ─────────────────────────────────────────────
        # observation_space: Box de uint8 [0, 255] con shape (H, W, C).
        # SB3 con CnnPolicy lo normaliza automáticamente a [0, 1].
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(IM_HEIGHT, IM_WIDTH, IM_CHANNELS),
            dtype=np.uint8
        )

        # action_space: Discrete(9) — 9 acciones posibles.
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        # ── Estado interno ───────────────────────────────────────
        self.show_preview = show_preview
        self.front_camera = None
        self.collision_flag = False
        self.lane_invasion_flag = False
        self.actor_list = []
        self.npc_vehicles = []   # Lista separada para NPCs (persisten entre episodios)
        self.step_count = 0

        # ── Logging de recompensas ───────────────────────────────
        self.global_step = 0
        self.reward_log = []  # Acumula componentes para promediar

        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"logs/reward_log_{timestamp}.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_step", "avg_r_speed", "avg_r_orientation",
                "avg_r_lane", "avg_r_collision", "avg_total"
            ])

        # ── Conexión a CARLA ─────────────────────────────────────
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # ── Activar modo síncrono ────────────────────────────────
        # En modo síncrono, la simulación SOLO avanza cuando llamamos
        # world.tick(). Esto es ESENCIAL para RL porque:
        #   1. Cada step tiene exactamente el mismo delta_t (determinismo)
        #   2. No hay frames perdidos entre la acción y la observación
        #   3. Las recompensas son consistentes y reproducibles
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)

        # ── Traffic Manager ──────────────────────────────────────
        # El Traffic Manager (TM) controla el comportamiento de los NPCs
        # con autopilot. DEBE estar en modo síncrono cuando el world lo está,
        # si no, los NPCs se quedan congelados sin moverse.
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(42)

        # Spawnear NPCs una sola vez (persisten entre episodios)
        self._spawn_npc_traffic()

    def _spawn_npc_traffic(self):
        """
        Spawnea vehículos NPC con autopilot por el mapa.

        Los NPCs se crean UNA sola vez al inicializar el env y persisten
        entre episodios (no se destruyen en reset). Esto es más eficiente
        que recrearlos cada episodio y hace el tráfico más realista.

        try_spawn_actor() intenta spawnear un actor en una posición. Si la
        posición está ocupada (ej: otro vehículo ya está ahí), retorna None
        en vez de lanzar una excepción. Por eso es preferible a spawn_actor()
        cuando spawneamos muchos actores en posiciones aleatorias.

        set_autopilot(True) le da el control del vehículo al Traffic Manager,
        que se encarga de que respeten semáforos, carriles, velocidad, etc.
        """
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        # Filtrar blueprints de vehículos que coincidan con nuestros modelos
        all_vehicle_bps = self.blueprint_library.filter('*vehicle*')
        npc_blueprints = []
        for bp in all_vehicle_bps:
            if any(model in bp.id for model in NPC_VEHICLE_MODELS):
                npc_blueprints.append(bp)

        # Si no se encontraron modelos específicos, usar todos los vehículos
        if not npc_blueprints:
            npc_blueprints = list(all_vehicle_bps)

        count = min(NUM_NPC_VEHICLES, len(spawn_points))
        spawned = 0

        for i in range(count):
            bp = random.choice(npc_blueprints)

            # Algunos blueprints tienen atributo 'color' que podemos randomizar
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            npc = self.world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                # set_autopilot(True) entrega el control al Traffic Manager
                npc.set_autopilot(True, self.traffic_manager.get_port())
                self.npc_vehicles.append(npc)
                spawned += 1

        print(f"=== NPCs spawneados: {spawned}/{count} ===")

        # Hacer unos ticks para que los NPCs arranquen a moverse
        for _ in range(10):
            self.world.tick()

    def _setup_vehicle(self):
        """Spawnea el vehículo en un punto aleatorio del mapa."""
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        spawn_point = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

    def _setup_camera(self):
        """
        Configura la cámara RGB frontal.
        Se attachea al vehículo con una posición relativa (x=adelante, z=arriba).
        El callback guarda cada frame en self.front_camera.
        """
        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IM_HEIGHT))
        cam_bp.set_attribute("fov", str(FOV))

        cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda img: self._process_image(img))

    def _setup_collision_sensor(self):
        """
        Sensor de colisión: dispara un callback cada vez que el vehículo
        choca con algo. Solo setea un flag (no acumula historial).
        """
        col_bp = self.blueprint_library.find("sensor.other.collision")
        col_transform = carla.Transform()
        self.collision_sensor = self.world.spawn_actor(
            col_bp, col_transform, attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _setup_lane_sensor(self):
        """
        Sensor de invasión de carril: detecta cuando el vehículo cruza
        las líneas de marcación de carril (lane markings).
        Funciona del lado del cliente usando datos de OpenDRIVE.
        """
        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))

    def _process_image(self, image):
        """
        Callback de la cámara. Convierte image.raw_data (bytes BGRA)
        a un array numpy RGB de shape (H, W, 3) con dtype uint8.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((IM_HEIGHT, IM_WIDTH, 4))  # BGRA
        self.front_camera = array[:, :, :3]               # Descarta Alpha → BGR
        # Nota: SB3 CnnPolicy acepta BGR o RGB indistintamente porque
        # la CNN aprende sus propios filtros.

    def _on_collision(self, event):
        self.collision_flag = True

    def _on_lane_invasion(self, event):
        self.lane_invasion_flag = True

    def _get_speed_kmh(self):
        """Calcula la velocidad del vehículo en km/h desde el vector 3D de velocidad."""
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def _compute_reward(self):
        """
        Función de recompensa por componentes.

        Componentes:
          1. r_speed: recompensa por mantener una velocidad cercana al target.
             Usa una curva gaussiana centrada en TARGET_SPEED_KMH.
             - Velocidad ideal → +R_SPEED_MAX
             - Muy lento o muy rápido → cercano a 0
          2. r_orientation: recompensa por estar alineado con la dirección de la ruta.
             Usa el producto punto entre el forward del vehículo y el forward del
             waypoint más cercano. Si apuntás en la misma dirección → +1.
          3. r_lane: penalidad por cruzar líneas de carril.
          4. r_collision: penalidad por colisionar.
          5. r_offroad: penalidad si el waypoint más cercano no es de tipo Driving.

        Retorna: (reward_total, dict_componentes, terminated)
        """
        speed = self._get_speed_kmh()
        terminated = False

        # ── 1. Recompensa por velocidad ──────────────────────────
        # math.exp(-x²) es una gaussiana: vale 1 cuando x=0 y decrece suavemente.
        # Dividimos por 20 para que la campana sea ancha (tolerante a ±20 km/h).
        speed_diff = speed - TARGET_SPEED_KMH
        r_speed = R_SPEED_MAX * math.exp(-(speed_diff / 20.0) ** 2)

        # ── 2. Recompensa por orientación ────────────────────────
        # get_waypoint() encuentra el waypoint más cercano a la ubicación del vehículo.
        # El waypoint tiene un transform que indica la dirección "ideal" de la ruta.
        # Comparamos la dirección del vehículo con la del waypoint usando producto punto.
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        waypoint = self.map.get_waypoint(vehicle_loc, project_to_road=True)

        # get_forward_vector() retorna un vector unitario en la dirección
        # en la que "mira" el actor/waypoint.
        veh_fwd = vehicle_transform.get_forward_vector()
        wp_fwd = waypoint.transform.get_forward_vector()

        # Producto punto entre vectores 2D (ignoramos z).
        # Si ambos apuntan igual → dot ≈ 1.0
        # Si son perpendiculares → dot ≈ 0.0
        # Si son opuestos → dot ≈ -1.0
        dot = veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y
        # Clamp a [0, 1]: no queremos recompensar ir en reversa
        r_orientation = R_ORIENTATION_MAX * max(0.0, dot)

        # ── 3. Penalidad por invasión de carril ──────────────────
        r_lane = R_LANE_PENALTY if self.lane_invasion_flag else 0.0
        self.lane_invasion_flag = False  # Reset para el próximo step

        # ── 4. Penalidad por colisión ────────────────────────────
        r_collision = 0.0
        if self.collision_flag:
            r_collision = R_COLLISION_PENALTY
            terminated = True  # Fin del episodio

        # ── 5. Penalidad por estar fuera de ruta ─────────────────
        # waypoint.lane_type indica el tipo de carril (Driving, Sidewalk, etc.)
        # Si el vehículo está en un carril que no es de conducción, penalizamos.
        r_offroad = 0.0
        if waypoint.lane_type != carla.LaneType.Driving:
            r_offroad = R_OFFROAD_PENALTY

        total = r_speed + r_orientation + r_lane + r_collision + r_offroad

        components = {
            "r_speed": r_speed,
            "r_orientation": r_orientation,
            "r_lane": r_lane,
            "r_collision": r_collision,
            "r_offroad": r_offroad,
        }

        return total, components, terminated

    def _log_rewards(self, components):
        """Acumula componentes y escribe promedios al CSV cada 100 steps."""
        self.reward_log.append(components)
        self.global_step += 1

        if self.global_step % 100 == 0 and self.reward_log:
            keys = ["r_speed", "r_orientation", "r_lane", "r_collision"]
            avgs = {k: np.mean([r[k] for r in self.reward_log]) for k in keys}
            avg_total = sum(avgs.values())

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.global_step,
                    f"{avgs['r_speed']:.4f}",
                    f"{avgs['r_orientation']:.4f}",
                    f"{avgs['r_lane']:.4f}",
                    f"{avgs['r_collision']:.4f}",
                    f"{avg_total:.4f}",
                ])
            self.reward_log = []

    # ══════════════════════════════════════════════════════════════
    #  INTERFAZ GYMNASIUM
    # ══════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        """
        Reinicia el episodio:
          1. Destruye todos los actores del episodio anterior
          2. Spawnea vehículo, cámara, sensores
          3. Espera al primer frame de cámara
          4. Retorna (observación, info) — formato Gymnasium v26+

        seed: SB3 puede pasar una seed para reproducibilidad.
        """
        super().reset(seed=seed)

        # Destruir actores previos
        for actor in reversed(self.actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list = []

        # Reset de flags
        self.collision_flag = False
        self.lane_invasion_flag = False
        self.front_camera = None
        self.step_count = 0
        self._ep_reward = 0  # Acumulador para el overlay de preview

        # Spawn de todo
        self._setup_vehicle()
        self._setup_camera()
        self._setup_collision_sensor()
        self._setup_lane_sensor()

        # Hacer algunos ticks para que CARLA procese los spawns
        # y la cámara envíe al menos un frame
        for _ in range(10):
            self.world.tick()

        # Esperar al primer frame (con timeout)
        wait = 0
        while self.front_camera is None and wait < 100:
            self.world.tick()
            wait += 1

        if self.front_camera is None:
            # Fallback: imagen negra si la cámara no respondió
            self.front_camera = np.zeros(
                (IM_HEIGHT, IM_WIDTH, IM_CHANNELS), dtype=np.uint8
            )

        return self.front_camera.copy(), {}

    def step(self, action):
        """
        Ejecuta una acción en el simulador.

        Parámetros:
            action (int): Índice de la acción discreta (0-8)

        Retorna (formato Gymnasium):
            observation: imagen RGB (H, W, 3) uint8
            reward: float con la recompensa total
            terminated: True si hubo colisión (episodio termina por condición del env)
            truncated: True si se alcanzó MAX_STEPS (episodio cortado por tiempo)
            info: dict con componentes de reward y velocidad
        """
        self.step_count += 1

        # ── Aplicar acción ───────────────────────────────────────
        steer, throttle, brake = DISCRETE_ACTIONS[action]
        # carla.VehicleControl envía los comandos al vehículo:
        #   throttle [0,1]: cuánto acelerar
        #   steer [-1,1]: dirección (-1 = izquierda total, 1 = derecha total)
        #   brake [0,1]: cuánto frenar
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

        # ── Avanzar simulación ───────────────────────────────────
        # world.tick() avanza exactamente FIXED_DELTA segundos en el simulador.
        self.world.tick()

        # ── Observación ──────────────────────────────────────────
        if self.front_camera is not None:
            obs = self.front_camera.copy()
        else:
            obs = np.zeros((IM_HEIGHT, IM_WIDTH, IM_CHANNELS), dtype=np.uint8)

        # ── Recompensa ───────────────────────────────────────────
        reward, components, terminated = self._compute_reward()

        # ── Truncado por tiempo ──────────────────────────────────
        truncated = self.step_count >= MAX_STEPS

        # ── Logging ──────────────────────────────────────────────
        self._log_rewards(components)

        # ── Preview ──────────────────────────────────────────────
        if self.show_preview:
            # cv2.resize escala la imagen de la cámara (160×120) a un tamaño
            # visible en pantalla (640×480). INTER_NEAREST mantiene los píxeles
            # nítidos en vez de hacer blur (útil para ver qué "ve" la CNN).
            display = cv2.resize(
                obs, (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_NEAREST
            )

            # Overlay con información en tiempo real sobre la imagen
            speed = self._get_speed_kmh()
            action_names = {
                0: "IZQ+Th", 1: "RECTO+Th", 2: "DER+Th",
                3: "IZQ+Full", 4: "RECTO+Full", 5: "DER+Full",
                6: "IZQ+Freno", 7: "RECTO+Freno", 8: "DER+Freno",
            }
            # cv2.putText dibuja texto sobre una imagen numpy.
            # Parámetros: imagen, texto, posición (x,y), font, escala, color, grosor
            info_text = f"Speed: {speed:.0f} km/h | Action: {action_names.get(action, '?')} | Step: {self.step_count}"
            cv2.putText(display, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            reward_text = f"Reward: {reward:+.2f} | Ep reward: {getattr(self, '_ep_reward', 0):+.1f}"
            cv2.putText(display, reward_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("CARLA Agent View", display)
            cv2.waitKey(1)

        # Acumular reward del episodio para el overlay
        self._ep_reward = getattr(self, '_ep_reward', 0) + reward

        info = {
            "speed_kmh": self._get_speed_kmh(),
            **components,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        # Destruir actores del episodio (vehículo ego, sensores)
        for actor in reversed(self.actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list = []

        # Destruir NPCs
        for npc in self.npc_vehicles:
            if npc is not None and npc.is_alive:
                npc.destroy()
        self.npc_vehicles = []
        print("=== NPCs destruidos ===")

        # Desactivar modo síncrono del Traffic Manager
        self.traffic_manager.set_synchronous_mode(False)

        # Restaurar settings originales (modo asíncrono)
        self.world.apply_settings(self.original_settings)

        if self.show_preview:
            cv2.destroyAllWindows()