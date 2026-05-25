"""
Entorno Gymnasium para CARLA — Modelo 3: Reward Multiplicativa Jerárquica.

Cambios vs Modelo 2 (solo reward; todo lo demás idéntico):
  - Se introduce r_safety_gate ∈ [0, 1] como producto de 4 factores suaves:
    g_orientation × g_lane × g_speed × g_direction.
  - r_progress y r_lane_center se MULTIPLICAN por el gate. Si la seguridad
    colapsa, el reward positivo colapsa con ella → no se puede compensar
    inseguridad con progreso (core de la hipótesis del paper).
  - r_speed queda FUERA del gate (señal mínima siempre presente).
  - Las penalidades terminales (colisión, offroad, lane, stall, wrong_way)
    NO se modulan: castigo pleno sin importar el gate.

Vector de observación: 10 valores (idéntico a M1/M2, NO se toca).
Acciones: 9 discretas (idéntico a M1/M2, NO se toca).
Hiperparámetros del PPO: idénticos a M1/M2 (comparación justa).
"""

import os, sys, glob, random, math, csv
from datetime import datetime
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# ═══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

IM_WIDTH, IM_HEIGHT, IM_CHANNELS, FOV = 640, 480, 3, 110
OBS_WIDTH, OBS_HEIGHT = 160, 120
VECTOR_OBS_SIZE = 10

FIXED_DELTA = 0.05
MAX_STEPS = 2000
TARGET_SPEED_KMH = 30

# ── Reward multiplicativa jerárquica (Modelo 3) ──
R_SPEED_MAX = 1.0                     # Fuera del gate (señal mínima siempre).
R_PROGRESS_PER_METER = 1.0            # Modulado por gate.
R_LANE_CENTER_MAX = 0.5               # Modulado por gate.
R_WRONG_WAY_PENALTY = -2.0            # Penalidad directa, no modulada.

# Parámetros del safety gate (escalas de tolerancia para cada factor).
GATE_LANE_TOLERANCE = 1.5             # metros: a 1.5m del centro, g_lane = 0.
GATE_SPEED_TOLERANCE = 25.0           # km/h: ancho de la gaussiana de velocidad.
GATE_ORIENT_SHARPNESS = 1.0           # exponente: más alto = más severo con desalineación.

# ── Penalidades (idénticas a M1/M2) ──
R_LANE_PENALTY = -5.0
R_COLLISION_PENALTY = -10.0
R_OFFROAD_PENALTY = -5.0

# ── Stall (idéntico a M1/M2) ──
R_STALL_PENALTY = -5.0
STALL_TERMINATE_STEPS = 30

# ── NPCs ──
NUM_NPC_VEHICLES = 20
NPC_VEHICLE_MODELS = [
    'dodge', 'audi', 'mini', 'mustang', 'lincoln',
    'prius', 'nissan', 'crown', 'impala',
]
MAX_SPAWN_RETRIES = 50

PREVIEW_WIDTH, PREVIEW_HEIGHT = 640, 480

DISCRETE_ACTIONS = {
    0: (-0.3, 0.6, 0.0),   1: ( 0.0, 0.6, 0.0),   2: ( 0.3, 0.6, 0.0),
    3: (-0.3, 1.0, 0.0),   4: ( 0.0, 1.0, 0.0),   5: ( 0.3, 1.0, 0.0),
    6: (-0.3, 0.0, 0.5),   7: ( 0.0, 0.0, 0.5),   8: ( 0.3, 0.0, 0.5),
}


class CarlaEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, show_preview=False):
        super().__init__()

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), np.uint8),
            "vector": spaces.Box(-1.0, 2.0, (VECTOR_OBS_SIZE,), np.float32),
        })
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        self.show_preview = show_preview
        self.front_camera = None
        self.collision_flag = False
        self.lane_invasion_flag = False
        self.actor_list = []
        self.npc_vehicles = []
        self.step_count = 0
        self.episode_count = 0
        self._ep_reward = 0.0

        self.stall_counter = 0
        self.prev_steer = 0.0
        self.prev_location = None

        # Logging: agrego el gate y sus 4 factores para poder graficarlos en TensorBoard.
        self.global_step = 0
        self._reward_accum = {
            'r_speed': 0.0, 'r_progress': 0.0, 'r_lane_center': 0.0,
            'r_wrong_way': 0.0, 'r_lane': 0.0, 'r_collision': 0.0,
            'r_offroad': 0.0, 'r_stall': 0.0, 'r_total': 0.0,
            'gate': 0.0, 'g_orient': 0.0, 'g_lane': 0.0,
            'g_speed': 0.0, 'g_direction': 0.0,
        }
        self._accum_count = 0
        self._log_freq = 50

        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/reward_log_M3_{ts}.csv"
        self._csv_file = open(self.log_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'global_step', 'episode', 'step_in_ep',
            'avg_r_speed', 'avg_r_progress', 'avg_r_lane_center',
            'avg_r_wrong_way', 'avg_r_lane', 'avg_r_collision',
            'avg_r_offroad', 'avg_r_stall', 'avg_r_total',
            'avg_gate', 'avg_g_orient', 'avg_g_lane', 'avg_g_speed', 'avg_g_direction',
            'speed_kmh',
        ])

        # Conexión a CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(42)
        self._spawn_npc_traffic()

    # ── NPCs ──

    def _spawn_npc_traffic(self):
        """Spawnea tráfico NPC con autopilot."""
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        all_bps = self.blueprint_library.filter('*vehicle*')
        npc_bps = [bp for bp in all_bps if any(m in bp.id for m in NPC_VEHICLE_MODELS)]
        if not npc_bps:
            npc_bps = list(all_bps)

        count = min(NUM_NPC_VEHICLES, len(spawn_points))
        spawned = 0
        for i in range(count):
            bp = random.choice(npc_bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
            npc = self.world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                npc.set_autopilot(True, self.traffic_manager.get_port())
                self.npc_vehicles.append(npc)
                spawned += 1
        print(f"[CarlaEnv] NPCs spawneados: {spawned}/{count}")
        for _ in range(10):
            self.world.tick()

    # ── Spawn y sensores ──

    def _spawn_vehicle(self):
        """Spawnea el vehículo ego con reintentos."""
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        for i in range(min(MAX_SPAWN_RETRIES, len(spawn_points))):
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[i])
            if vehicle is not None:
                self.vehicle = vehicle
                self.actor_list.append(self.vehicle)
                return
        raise RuntimeError("No se pudo spawnear el vehículo.")

    def _setup_camera(self):
        """Configura cámara RGB frontal."""
        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IM_HEIGHT))
        cam_bp.set_attribute("fov", str(FOV))
        cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda img: self._process_image(img))

    def _setup_collision_sensor(self):
        """Configura sensor de colisión."""
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _setup_lane_sensor(self):
        """Configura sensor de invasión de carril."""
        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))

    # ── Callbacks ──

    def _process_image(self, image):
        """Convierte imagen CARLA BGRA 640×480 a BGR 160×120."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))
        self.front_camera = cv2.resize(array[:, :, :3], (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)

    def _on_collision(self, event):
        self.collision_flag = True

    def _on_lane_invasion(self, event):
        self.lane_invasion_flag = True

    # ── Utilidades ──

    def _get_speed_kmh(self):
        """Velocidad escalar en km/h."""
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def _is_wrong_way(self, wp_fwd):
        """Detecta contramano comparando velocidad con dirección del carril."""
        vel = self.vehicle.get_velocity()
        speed_vec = math.sqrt(vel.x**2 + vel.y**2)
        if speed_vec < 0.5:
            return False, 0.0
        vel_dot = (vel.x / speed_vec) * wp_fwd.x + (vel.y / speed_vec) * wp_fwd.y
        return vel_dot < 0.0, vel_dot

    def _get_distance_traveled(self):
        """Distancia euclidiana 2D desde el step anterior."""
        loc = self.vehicle.get_transform().location
        if self.prev_location is None:
            self.prev_location = loc
            return 0.0
        dist = math.sqrt((loc.x - self.prev_location.x)**2 + (loc.y - self.prev_location.y)**2)
        self.prev_location = loc
        return dist

    # ── Observación ──

    def _get_vector_obs(self):
        """Vector de 10 mediciones normalizadas (IDÉNTICO a M1/M2)."""
        speed = self._get_speed_kmh()
        vt = self.vehicle.get_transform()
        wp = self.map.get_waypoint(vt.location, project_to_road=True)
        veh_fwd = vt.get_forward_vector()
        wp_fwd = wp.transform.get_forward_vector()

        orientation = veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y
        dist_to_center = vt.location.distance(wp.transform.location)
        veh_yaw = math.atan2(veh_fwd.y, veh_fwd.x)
        wp_yaw = math.atan2(wp_fwd.y, wp_fwd.x)
        angle_diff = math.atan2(math.sin(veh_yaw - wp_yaw), math.cos(veh_yaw - wp_yaw))

        is_wrong, vel_dot = self._is_wrong_way(wp_fwd)

        return np.array([
            np.clip(speed / TARGET_SPEED_KMH, 0.0, 2.0),
            1.0 if self.collision_flag else 0.0,
            orientation,
            np.clip(dist_to_center / 3.0, 0.0, 1.0),
            angle_diff / math.pi,
            0.0 if wp.lane_type == carla.LaneType.Driving else 1.0,
            np.clip(self.stall_counter / STALL_TERMINATE_STEPS, 0.0, 1.0),
            1.0 if is_wrong else 0.0,
            vel_dot,
            np.clip(self.prev_steer, -1.0, 1.0),
        ], dtype=np.float32)

    def _build_obs(self):
        """Construye observación Dict {image, vector}."""
        img = self.front_camera.copy() if self.front_camera is not None else np.zeros((OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), dtype=np.uint8)
        return {"image": img, "vector": self._get_vector_obs()}

    # ── Safety Gate (Modelo 3) ──

    def _compute_safety_gate(self, speed, vel_dot, dist_to_center, orientation, is_wrong):
        """
        Calcula el gate de seguridad como producto de 4 factores ∈ [0, 1].
        El gate modula r_progress y r_lane_center: si cualquier factor cae a 0,
        todo el reward positivo modulado se anula.

        Factores:
          g_orientation: alineación vehículo-carril (orientation = dot de forwards).
            (orientation+1)/2 mapea [-1,1] → [0,1]. Elevado a GATE_ORIENT_SHARPNESS.
          g_lane: cercanía al centro del carril. Lineal: 1 en centro, 0 a tolerance.
          g_speed: campana gaussiana centrada en TARGET_SPEED. Penaliza ir muy lento
            o muy rápido.
          g_direction: 1 si va en sentido correcto, 0 si va en contramano.
            Usa vel_dot (sentido REAL de marcha), no orientation (forward del auto).
        """
        # g_orientation: penaliza desalineación del vehículo con el carril.
        # math.pow(x, n) = x^n. Con sharpness=1.0 es lineal; >1.0 castiga más fuerte
        # cualquier desalineación pequeña.
        g_orient = math.pow(max(0.0, (orientation + 1.0) / 2.0), GATE_ORIENT_SHARPNESS)

        # g_lane: 1 en el centro, decae lineal a 0 a GATE_LANE_TOLERANCE metros.
        g_lane = max(0.0, 1.0 - dist_to_center / GATE_LANE_TOLERANCE)

        # g_speed: campana gaussiana e^(-((v-target)/σ)²), vale 1 en TARGET_SPEED.
        # Penaliza tanto velocidades muy bajas como muy altas.
        g_speed = math.exp(-((speed - TARGET_SPEED_KMH) / GATE_SPEED_TOLERANCE) ** 2)

        # g_direction: corte duro (es la única condición binaria del gate).
        # Si va en contramano, todo el reward modulado se anula.
        # Detenido (vel_dot ≈ 0): vale 1, no penaliza estar quieto vía gate
        # (la quietud se castiga vía r_stall).
        g_direction = 0.0 if is_wrong else 1.0

        gate = g_orient * g_lane * g_speed * g_direction
        return gate, g_orient, g_lane, g_speed, g_direction

    # ── Reward (Modelo 3) ──

    def _compute_reward(self, action):
        """
        Modelo 3 — Reward multiplicativa jerárquica.

        r_total = r_safety_gate × (r_progress + r_lane_center)
                + r_speed
                + r_wrong_way + r_lane + r_collision + r_offroad + r_stall

        El gate modula solo las recompensas positivas de progreso/centrado.
        r_speed queda fuera (señal mínima siempre presente).
        Las penalidades terminales no se modulan: castigo pleno siempre.
        """
        speed = self._get_speed_kmh()
        terminated = False

        vt = self.vehicle.get_transform()
        wp = self.map.get_waypoint(vt.location, project_to_road=True)
        wp_fwd = wp.transform.get_forward_vector()
        veh_fwd = vt.get_forward_vector()

        # Métricas base (idénticas a M2).
        is_wrong, vel_dot = self._is_wrong_way(wp_fwd)
        vel_dot = float(np.clip(vel_dot, -1.0, 1.0))
        orientation = veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y
        dist_to_center = vt.location.distance(wp.transform.location)

        # ── SAFETY GATE ──
        gate, g_orient, g_lane_factor, g_speed, g_direction = self._compute_safety_gate(
            speed, vel_dot, dist_to_center, orientation, is_wrong
        )

        # ── Componentes positivas ──

        # r_speed: gaussiana centrada en TARGET_SPEED. FUERA del gate.
        # math.exp(x) = e^x. Forma e^(-((x-μ)/σ)²) → campana que vale 1 en μ.
        r_speed = R_SPEED_MAX * math.exp(-((speed - TARGET_SPEED_KMH) / 20.0) ** 2)

        # r_progress: distancia recorrida (escalar). Será modulada por gate.
        # Diferencia clave vs M2: en M2 multiplicamos por vel_dot para dar sentido;
        # acá no hace falta porque g_direction ya anula todo si va contramano.
        dist = self._get_distance_traveled()
        r_progress_raw = R_PROGRESS_PER_METER * dist

        # r_lane_center: señal densa, máxima en el centro. Será modulada por gate.
        r_lane_center_raw = R_LANE_CENTER_MAX * max(0.0, 1.0 - dist_to_center / 1.5)

        # ── APLICACIÓN DEL GATE ──
        # Solo modulamos las componentes que dependen de "estar haciendo bien las cosas".
        # Si el gate cae a 0 (mal alineado, descentrado, muy lento/rápido o contramano),
        # estas componentes valen 0 sin importar el progreso real.
        r_progress = gate * r_progress_raw
        r_lane_center = gate * r_lane_center_raw

        # ── Componentes negativas (NO se modulan) ──

        # r_wrong_way: penalidad directa por ir en contramano.
        # Aunque g_direction ya anula el reward positivo, mantenemos la penalidad
        # explícita para que la señal sea aún más fuerte (gradient claro hacia "no".
        r_wrong_way = R_WRONG_WAY_PENALTY if is_wrong else 0.0

        # Steer tracking (igual a M1/M2).
        steer, _, _ = DISCRETE_ACTIONS[action]
        self.prev_steer = steer

        # Stall (igual a M1/M2).
        if speed < 2.0:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        r_stall = 0.0
        if self.stall_counter >= STALL_TERMINATE_STEPS:
            r_stall = R_STALL_PENALTY
            terminated = True

        # Lane invasion (igual a M1/M2).
        r_lane = R_LANE_PENALTY if self.lane_invasion_flag else 0.0
        self.lane_invasion_flag = False

        # Colisión (igual a M1/M2).
        r_collision = 0.0
        if self.collision_flag:
            r_collision = R_COLLISION_PENALTY
            terminated = True

        # Offroad (igual a M1/M2).
        r_offroad = R_OFFROAD_PENALTY if wp.lane_type != carla.LaneType.Driving else 0.0

        total = (r_speed + r_progress + r_lane_center + r_wrong_way
                 + r_stall + r_lane + r_collision + r_offroad)

        components = {
            'r_speed': r_speed, 'r_progress': r_progress, 'r_lane_center': r_lane_center,
            'r_wrong_way': r_wrong_way, 'r_stall': r_stall,
            'r_lane': r_lane, 'r_collision': r_collision, 'r_offroad': r_offroad,
            'gate': gate, 'g_orient': g_orient, 'g_lane': g_lane_factor,
            'g_speed': g_speed, 'g_direction': g_direction,
        }
        return total, components, terminated

    # ── Logging ──

    def _log_rewards(self, components, total):
        """Acumula y escribe promedios al CSV cada _log_freq steps."""
        for k, v in components.items():
            self._reward_accum[k] += v
        self._reward_accum['r_total'] += total
        self._accum_count += 1

        if self._accum_count >= self._log_freq:
            n = self._accum_count
            speed = self._get_speed_kmh()
            self._csv_writer.writerow([
                self.global_step, self.episode_count, self.step_count,
                *[round(self._reward_accum[k] / n, 4) for k in [
                    'r_speed', 'r_progress', 'r_lane_center',
                    'r_wrong_way', 'r_lane', 'r_collision', 'r_offroad',
                    'r_stall', 'r_total',
                    'gate', 'g_orient', 'g_lane', 'g_speed', 'g_direction']],
                round(speed, 2),
            ])
            self._csv_file.flush()
            for k in self._reward_accum:
                self._reward_accum[k] = 0.0
            self._accum_count = 0

    # ── Interfaz Gymnasium ──

    def reset(self, seed=None, options=None):
        """Reinicia episodio: destruye actores, spawnea vehículo y sensores."""
        super().reset(seed=seed)
        for actor in reversed(self.actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list = []

        self.collision_flag = False
        self.lane_invasion_flag = False
        self.front_camera = None
        self.step_count = 0
        self.stall_counter = 0
        self.prev_steer = 0.0
        self.prev_location = None
        self._ep_reward = 0.0
        self.episode_count += 1
        for k in self._reward_accum:
            self._reward_accum[k] = 0.0
        self._accum_count = 0

        self._spawn_vehicle()
        self._setup_camera()
        self._setup_collision_sensor()
        self._setup_lane_sensor()

        for _ in range(10):
            self.world.tick()
        wait = 0
        while self.front_camera is None and wait < 100:
            self.world.tick()
            wait += 1
        if self.front_camera is None:
            self.front_camera = np.zeros((OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), dtype=np.uint8)

        return self._build_obs(), {}

    def step(self, action):
        """Ejecuta acción, calcula reward, retorna (obs, reward, terminated, truncated, info)."""
        self.step_count += 1
        self.global_step += 1

        steer, throttle, brake = DISCRETE_ACTIONS[action]
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self.world.tick()

        reward, components, terminated = self._compute_reward(action)
        obs = self._build_obs()
        truncated = self.step_count >= MAX_STEPS

        self._log_rewards(components, reward)
        self._ep_reward += reward

        if self.show_preview:
            self._render_preview(obs["image"], action, reward, components)

        return obs, reward, terminated, truncated, {"speed_kmh": self._get_speed_kmh(), **components}

    # ── Preview ──

    def _render_preview(self, obs_image, action, reward, components):
        """Muestra imagen con overlay informativo."""
        display = cv2.resize(obs_image, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_NEAREST)
        speed = self._get_speed_kmh()
        names = {0:"IZQ+Th",1:"RECTO+Th",2:"DER+Th",3:"IZQ+Full",4:"RECTO+Full",5:"DER+Full",6:"IZQ+Freno",7:"RECTO+Freno",8:"DER+Freno"}
        cv2.putText(display, f"Speed:{speed:.0f} | {names.get(action,'?')} | Step:{self.step_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(display, f"R:{reward:+.2f} | Gate:{components['gate']:.2f} | Ep:{self._ep_reward:+.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("CARLA Agent View - M3", display)
        cv2.waitKey(1)

    # ── Limpieza ──

    def close(self):
        """Libera recursos: CSV, actores, NPCs, restaura settings."""
        if hasattr(self, '_csv_file') and not self._csv_file.closed:
            self._csv_file.close()
            print(f"[CarlaEnv] Log: {self.log_path}")
        for actor in reversed(self.actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list = []
        for npc in self.npc_vehicles:
            if npc is not None and npc.is_alive:
                npc.destroy()
        self.npc_vehicles = []
        try: self.traffic_manager.set_synchronous_mode(False)
        except: pass
        try: self.world.apply_settings(self._original_settings)
        except: pass
        if self.show_preview:
            cv2.destroyAllWindows()