"""
Entorno Gymnasium para CARLA con PPO + MultiInputPolicy (Stable Baselines3).

Características principales:
  - Modo síncrono con fixed_delta_seconds (determinismo para RL)
  - Acciones discretas: 9 combinaciones de steer × throttle/brake
  - Observación Dict: imagen RGB + vector auxiliar con mediciones
  - Función de recompensa por componentes: velocidad, orientación, carril, colisión, offroad
  - Spawn con try_spawn_actor() y reintentos
  - Tráfico NPC persistente entre episodios
  - Logging de recompensas a CSV para diagnóstico
  - Preview opcional con overlay de información en tiempo real
"""

import os
import sys
import glob
import random
import math
import csv
import time
from datetime import datetime

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════

# Resolución del sensor CARLA
IM_WIDTH = 640          
IM_HEIGHT = 480        
IM_CHANNELS = 3         
FOV = 110    

# Resolución de la observación
OBS_WIDTH = 160
OBS_HEIGHT = 120

# [speed, collision, orientation, distance_to_center, angle_diff, is_offroad, stall_ratio]
VECTOR_OBS_SIZE = 7

FIXED_DELTA = 0.05      
MAX_STEPS = 2000       

TARGET_SPEED_KMH = 30

R_SPEED_MAX = 1.0            
R_ORIENTATION_MAX = 1.0      
R_LANE_PENALTY = -5.0         
R_COLLISION_PENALTY = -10.0   
R_OFFROAD_PENALTY = -5.0      

NUM_NPC_VEHICLES = 20
NPC_VEHICLE_MODELS = [
    'dodge', 'audi', 'mini', 'mustang', 'lincoln',
    'prius', 'nissan', 'crown', 'impala',
]

MAX_SPAWN_RETRIES = 50

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

DISCRETE_ACTIONS = {
    0: (-0.3, 0.6, 0.0),   # Izquierda + aceleración media
    1: ( 0.0, 0.6, 0.0),   # Recto + aceleración media
    2: ( 0.3, 0.6, 0.0),   # Derecha + aceleración media
    3: (-0.3, 1.0, 0.0),   # Izquierda + aceleración completa
    4: ( 0.0, 1.0, 0.0),   # Recto + aceleración completa
    5: ( 0.3, 1.0, 0.0),   # Derecha + aceleración completa
    6: (-0.3, 0.0, 0.5),   # Izquierda + freno
    7: ( 0.0, 0.0, 0.5),   # Recto + freno
    8: ( 0.3, 0.0, 0.5),   # Derecha + freno
}

class CarlaEnv(gym.Env):

    # Implementa la interfaz estándar de Gymnasium con observación Dict

    metadata = {"render_modes": ["human"]}

    def __init__(self, show_preview=False):
        super().__init__()

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS),
                dtype=np.uint8,
            ),
            "vector": spaces.Box(
                low=-1.0, high=2.0,
                shape=(VECTOR_OBS_SIZE,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        self.show_preview = show_preview
        self.front_camera = None
        self.collision_flag = False
        self.lane_invasion_flag = False
        self.actor_list = []       # Actores del episodio (vehículo ego + sensores)
        self.npc_vehicles = []     # NPCs persistentes entre episodios
        self.step_count = 0
        self.episode_count = 0
        self._ep_reward = 0.0

        self.global_step = 0
        self._reward_accum = {
            'r_speed': 0.0, 'r_orientation': 0.0,
            'r_progress': 0.0, 'r_stall': 0.0,
            'r_lane': 0.0, 'r_collision': 0.0, 'r_offroad': 0.0,
            'r_total': 0.0,
        }
        self._accum_count = 0
        self._log_freq = 50 

        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/reward_log_{timestamp}.csv"
        self._csv_file = open(self.log_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'global_step', 'episode', 'step_in_ep',
            'avg_r_speed', 'avg_r_orientation', 'avg_r_progress', 'avg_r_stall',
            'avg_r_lane', 'avg_r_collision', 'avg_r_offroad',
            'avg_r_total', 'speed_kmh',
        ])

        # Conexión a CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # Modo síncrono
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(42)

        self._spawn_npc_traffic()

    def _spawn_npc_traffic(self):
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        all_vehicle_bps = self.blueprint_library.filter('*vehicle*')
        npc_blueprints = [
            bp for bp in all_vehicle_bps
            if any(model in bp.id for model in NPC_VEHICLE_MODELS)
        ]
        if not npc_blueprints:
            npc_blueprints = list(all_vehicle_bps)

        count = min(NUM_NPC_VEHICLES, len(spawn_points))
        spawned = 0

        for i in range(count):
            bp = random.choice(npc_blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            npc = self.world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                npc.set_autopilot(True, self.traffic_manager.get_port())
                self.npc_vehicles.append(npc)
                spawned += 1

        print(f"[CarlaEnv] NPCs spawneados: {spawned}/{count}")

        for _ in range(10):
            self.world.tick()

    def _spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        retries = min(MAX_SPAWN_RETRIES, len(spawn_points))
        for i in range(retries):
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[i])
            if vehicle is not None:
                self.vehicle = vehicle
                self.actor_list.append(self.vehicle)
                return

        raise RuntimeError(
            f"No se pudo spawnear el vehículo después de {retries} intentos. "
            "Posiblemente hay demasiados actores ocupando los spawn points."
        )

    def _setup_camera(self):
        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IM_HEIGHT))
        cam_bp.set_attribute("fov", str(FOV))

        cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        self.camera = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=self.vehicle
        )
        self.actor_list.append(self.camera)
        self.camera.listen(lambda img: self._process_image(img))

    def _setup_collision_sensor(self):
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _setup_lane_sensor(self):
        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((IM_HEIGHT, IM_WIDTH, 4))       # BGRA a 640×480
        bgr = array[:, :, :3]                                  # Quitar canal alpha
        # cv2.resize(src, (ancho, alto)) — redimensiona la imagen al tamaño destino
        # INTER_AREA es el mejor método para reducir tamaño (promedia píxeles vecinos)
        self.front_camera = cv2.resize(bgr, (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)

    def _on_collision(self, event):
        self.collision_flag = True

    def _on_lane_invasion(self, event):
        self.lane_invasion_flag = True

    def _get_speed_kmh(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def _get_vector_obs(self):
        """
        Construye el vector auxiliar de mediciones normalizadas.
        """
        speed = self._get_speed_kmh()
        vehicle_transform = self.vehicle.get_transform()
        waypoint = self.map.get_waypoint(
            vehicle_transform.location, project_to_road=True
        )

        # Velocidad normalizada
        speed_norm = np.clip(speed / TARGET_SPEED_KMH, 0.0, 2.0)

        # Flag de colisión (binario)
        collision = 1.0 if self.collision_flag else 0.0

        # Orientación relativa al carril
        veh_fwd = vehicle_transform.get_forward_vector()
        wp_fwd = waypoint.transform.get_forward_vector()
        orientation = veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y

        # Distancia lateral al centro del carril
        dist_to_center = vehicle_transform.location.distance(waypoint.transform.location)
        dist_to_center_norm = np.clip(dist_to_center / 3.0, 0.0, 1.0)

        # Diferencia angular entre vehículo y carril
        veh_yaw = math.atan2(veh_fwd.y, veh_fwd.x)
        wp_yaw = math.atan2(wp_fwd.y, wp_fwd.x)
        angle_diff = veh_yaw - wp_yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        angle_diff_norm = angle_diff / math.pi

        # Flag de offroad
        is_offroad = 0.0 if waypoint.lane_type == carla.LaneType.Driving else 1.0

        # Ratio de stall
        stall_ratio = np.clip(self.stall_steps / 30.0, 0.0, 1.0)

        return np.array([
            speed_norm,
            collision,
            orientation,
            dist_to_center_norm,
            angle_diff_norm,
            is_offroad,
            stall_ratio,
        ], dtype=np.float32)

    def _build_obs(self):
        """
        Construye la observación completa como un dict con "image" y "vector".
        Este es el formato que espera MultiInputPolicy de SB3.
        """
        if self.front_camera is not None:
            image = self.front_camera.copy()
        else:
            image = np.zeros((OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), dtype=np.uint8)

        vector = self._get_vector_obs()

        return {"image": image, "vector": vector}

    # Función de recompensa
    def _compute_reward(self):
        """
        Componentes:
        1. r_speed: gaussiana centrada en TARGET_SPEED_KMH.
            Si el auto está casi parado (< 2 km/h), penalidad directa fuerte (-5.0).
        2. r_orientation: producto punto entre el forward vector del
            vehículo y el del waypoint más cercano.
        3. r_progress: incentivo de distancia recorrida entre pasos.
        4. r_lane: penalidad por invasión de carril.
        5. r_collision: penalidad por colisión y fin de episodio.
        6. r_offroad: penalidad por salir del carril de conducción.

        Retorna: (reward_total, dict_componentes, terminated)
        """
        speed = self._get_speed_kmh()
        terminated = False

        # 1. Recompensa por velocidad (gaussiana + penalidad por quietud)
        speed_diff = speed - TARGET_SPEED_KMH
        if speed < 2.0:
            r_speed = -5.0
            self.stall_steps += 1
        else:
            r_speed = R_SPEED_MAX * math.exp(-(speed_diff / 20.0) ** 2)
            self.stall_steps = 0

        # 2. Recompensa por orientación (solo si va a una velocidad significativa)
        vehicle_transform = self.vehicle.get_transform()
        waypoint = self.map.get_waypoint(
            vehicle_transform.location, project_to_road=True
        )
        veh_fwd = vehicle_transform.get_forward_vector()
        wp_fwd = waypoint.transform.get_forward_vector()

        dot = veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y

        if speed > 5.0:
            r_orientation = R_ORIENTATION_MAX * max(0.0, dot)
        else:
            r_orientation = 0.0

        # 3. Recompensa por progreso (distancia al anterior frame)
        r_progress = 0.0
        current_loc = vehicle_transform.location
        if self.prev_location is not None:
            dx = current_loc.distance(self.prev_location)
            r_progress = 1.0 * min(dx, 5.0)
        self.prev_location = current_loc

        # 4. Penalidad por invasión de carril
        r_lane = R_LANE_PENALTY if self.lane_invasion_flag else 0.0
        self.lane_invasion_flag = False

        # 5. Penalidad por colisión
        r_collision = 0.0
        if self.collision_flag:
            r_collision = R_COLLISION_PENALTY
            terminated = True

        # 6. Penalidad por offroad
        r_offroad = 0.0
        if waypoint.lane_type != carla.LaneType.Driving:
            r_offroad = R_OFFROAD_PENALTY

        # 7. Corte por stall prolongado
        r_stall = 0.0
        if self.stall_steps >= 30:
            terminated = True
            r_stall = -3.0

        total = (
            r_speed + r_orientation + r_progress + r_lane
            + r_collision + r_offroad + r_stall
        )

        components = {
            'r_speed': r_speed,
            'r_orientation': r_orientation,
            'r_progress': r_progress,
            'r_stall': r_stall,
            'r_lane': r_lane,
            'r_collision': r_collision,
            'r_offroad': r_offroad,
        }
        return total, components, terminated

    def _log_rewards(self, components, total):
        """
        Acumula componentes de recompensa y escribe promedios al CSV
        cada _log_freq steps.
        """
        for key, val in components.items():
            self._reward_accum[key] += val
        self._reward_accum['r_total'] += total
        self._accum_count += 1

        if self._accum_count >= self._log_freq:
            n = self._accum_count
            speed = self._get_speed_kmh()

            self._csv_writer.writerow([
                self.global_step, self.episode_count, self.step_count,
                round(self._reward_accum['r_speed'] / n, 4),
                round(self._reward_accum['r_orientation'] / n, 4),
                round(self._reward_accum['r_progress'] / n, 4),
                round(self._reward_accum['r_stall'] / n, 4),
                round(self._reward_accum['r_lane'] / n, 4),
                round(self._reward_accum['r_collision'] / n, 4),
                round(self._reward_accum['r_offroad'] / n, 4),
                round(self._reward_accum['r_total'] / n, 4),
                round(speed, 2),
            ])
            self._csv_file.flush()

            for key in self._reward_accum:
                self._reward_accum[key] = 0.0
            self._accum_count = 0

    def reset(self, seed=None, options=None):
        """
        Reinicia el episodio:
          1. Destruye actores del episodio anterior (vehículo ego + sensores)
          2. Spawnea vehículo, cámara y sensores
          3. Espera al primer frame de cámara
          4. Retorna (observación_dict, info) — formato Gymnasium v26+
        """
        super().reset(seed=seed)

        # Destruir actores del episodio anterior
        for actor in reversed(self.actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list = []

        # Reset de flags y contadores
        self.collision_flag = False
        self.lane_invasion_flag = False
        self.front_camera = None
        self.step_count = 0
        self.stall_steps = 0
        self.prev_location = None
        self._ep_reward = 0.0
        self.episode_count += 1

        # Reset acumuladores de logging
        for key in self._reward_accum:
            self._reward_accum[key] = 0.0
        self._accum_count = 0

        # Spawn de vehículo y sensores
        self._spawn_vehicle()
        self._setup_camera()
        self._setup_collision_sensor()
        self._setup_lane_sensor()

        # Hacer algunos ticks para que CARLA procese los spawns
        for _ in range(10):
            self.world.tick()

        # Esperar primer frame de cámara (con timeout)
        wait = 0
        while self.front_camera is None and wait < 100:
            self.world.tick()
            wait += 1

        if self.front_camera is None:
            self.front_camera = np.zeros(
                (OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), dtype=np.uint8
            )

        return self._build_obs(), {}

    def step(self, action):
        """
        Ejecuta una acción discreta en el simulador.

        Retorna (formato Gymnasium):
            observation: dict con "image" y "vector"
            reward: float — recompensa total del step
            terminated: True si hubo colisión o stall
            truncated: True si se alcanzó MAX_STEPS
            info: dict con componentes de reward y velocidad
        """
        self.step_count += 1
        self.global_step += 1

        # 1. Aplicar acción
        steer, throttle, brake = DISCRETE_ACTIONS[action]
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

        # 2. Avanzar simulación
        self.world.tick()

        # 3. Calcular recompensa (antes de construir obs para que stall_steps esté actualizado)
        reward, components, terminated = self._compute_reward()

        # 4. Construir observación dict
        obs = self._build_obs()

        # 5. Verificar truncado por tiempo
        truncated = self.step_count >= MAX_STEPS

        # 6. Logging
        self._log_rewards(components, reward)
        self._ep_reward += reward

        # 7. Preview (opcional)
        if self.show_preview:
            self._render_preview(obs["image"], action, reward)

        info = {
            "speed_kmh": self._get_speed_kmh(),
            **components,
        }
        return obs, reward, terminated, truncated, info


    def _render_preview(self, obs_image, action, reward):
        """
        Muestra la imagen de la cámara escalada con overlay de información.
        """
        display = cv2.resize(
            obs_image, (PREVIEW_WIDTH, PREVIEW_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )

        speed = self._get_speed_kmh()
        action_names = {
            0: "IZQ+Th", 1: "RECTO+Th", 2: "DER+Th",
            3: "IZQ+Full", 4: "RECTO+Full", 5: "DER+Full",
            6: "IZQ+Freno", 7: "RECTO+Freno", 8: "DER+Freno",
        }

        info_text = (
            f"Speed: {speed:.0f} km/h | "
            f"Action: {action_names.get(action, '?')} | "
            f"Step: {self.step_count}"
        )
        cv2.putText(
            display, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        reward_text = f"Reward: {reward:+.2f} | Ep total: {self._ep_reward:+.1f}"
        cv2.putText(
            display, reward_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        cv2.imshow("CARLA Agent View", display)
        cv2.waitKey(1)

    # ──────────────────────────────────────────────────────────────
    #  LIMPIEZA
    # ──────────────────────────────────────────────────────────────

    def close(self):
        """
        Libera todos los recursos:
          1. Cierra el archivo CSV de logging
          2. Destruye actores del episodio actual (vehículo ego + sensores)
          3. Destruye NPCs
          4. Desactiva modo síncrono del Traffic Manager
          5. Restaura los settings originales del world (modo asíncrono)
          6. Cierra ventanas de cv2 si hay preview
        """
        # CSV
        if hasattr(self, '_csv_file') and not self._csv_file.closed:
            self._csv_file.close()
            print(f"[CarlaEnv] Log guardado en: {self.log_path}")

        # Actores del episodio
        for actor in reversed(self.actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list = []

        # NPCs
        for npc in self.npc_vehicles:
            if npc is not None and npc.is_alive:
                npc.destroy()
        self.npc_vehicles = []
        print("[CarlaEnv] NPCs destruidos")

        # Restaurar Traffic Manager y settings
        try:
            self.traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

        try:
            self.world.apply_settings(self._original_settings)
        except Exception:
            pass

        if self.show_preview:
            cv2.destroyAllWindows()