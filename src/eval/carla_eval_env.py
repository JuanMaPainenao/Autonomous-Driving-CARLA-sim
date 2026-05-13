"""
Entorno de evaluación para CARLA. NO computa reward (no se necesita para
evaluar) — solo loggea telemetría detallada para análisis posterior.

Diferencias clave vs el env de entrenamiento:
  - Permite cargar town arbitrario (Town10HD o Town02) vía load_world().
  - NPCs opcionales (flag with_npcs).
  - Spawn points seleccionados con seed fija → mismo escenario para los 3 modelos.
  - Telemetría por step: posición XY, velocidad, dist_to_center, orientation,
    vel_dot, is_wrong_way, action, lane_invasion → análisis de hipótesis.
  - El observation_space y action_space son IDÉNTICOS al training env, para
    que PPO.load() funcione sin problemas.
"""

import os, sys, glob, random, math, time
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

from eval_configs import (
    IM_WIDTH, IM_HEIGHT, IM_CHANNELS, FOV, OBS_WIDTH, OBS_HEIGHT,
    VECTOR_OBS_SIZE, FIXED_DELTA, MAX_STEPS, TARGET_SPEED_KMH,
    STALL_TERMINATE_STEPS, DISCRETE_ACTIONS,
    NUM_NPC_VEHICLES, NPC_VEHICLE_MODELS, CLIENT_TIMEOUT,
)


class CarlaEvalEnv(gym.Env):
    """Entorno Gymnasium para evaluación. No computa reward."""

    metadata = {"render_modes": []}

    def __init__(self, town="Town10HD", with_npcs=True, seed=42):
        super().__init__()
        self.town = town
        self.with_npcs = with_npcs
        self.seed_val = seed

        # Observation/action space IDÉNTICOS al training env.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), np.uint8),
            "vector": spaces.Box(-1.0, 2.0, (VECTOR_OBS_SIZE,), np.float32),
        })
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        # Estado interno
        self.front_camera = None
        self.collision_flag = False
        self.lane_invasion_flag = False
        self.actor_list = []
        self.npc_vehicles = []
        self.step_count = 0
        self.episode_count = 0

        self.stall_counter = 0
        self.prev_steer = 0.0
        self.prev_location = None

        # Telemetría del episodio actual (se reinicia en cada reset).
        self.episode_telemetry = []

        # Conexión a CARLA con timeout alto para load_world.
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(CLIENT_TIMEOUT)

        # Carga del mapa con detección inteligente:
        # Si CARLA ya está corriendo el town deseado, NO hacemos load_world
        # (evita segfaults en GPUs con poca VRAM al hacer swap entre towns).
        # Comparamos con `in` porque CARLA puede reportar el nombre con
        # sufijo `_Opt` (ej: "Town10HD_Opt") aunque pidamos "Town10HD".
        current_world = self.client.get_world()
        current_map_name = current_world.get_map().name
        if self.town in current_map_name:
            print(f"[EvalEnv] {self.town} ya está cargado (actual: {current_map_name}) — no se recarga.")
            self.world = current_world
        else:
            print(f"[EvalEnv] Cargando {self.town} (actual: {current_map_name})...")
            # load_world() sin reset_settings: pattern del código de prueba
            # que funciona consistentemente. CARLA tarda 30-90s en cargar
            # el mundo nuevo; el timeout de 120s del cliente cubre eso.
            self.world = self.client.load_world(self.town)
            # Pausa breve para que CARLA termine de inicializar el mundo
            # antes de pedirle settings/blueprint_library/etc.
            time.sleep(2.0)
            print(f"[EvalEnv] {self.town} cargado.")

        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # Sincronía + fixed delta (idéntico al training).
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)

        # Traffic manager con la MISMA seed que el seed del env → mismo
        # comportamiento de NPCs entre corridas de M1/M2/M3.
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.seed_val)

        # Seed de Python para spawn points y blueprints aleatorios.
        random.seed(self.seed_val)

        if self.with_npcs:
            self._spawn_npc_traffic()

        # Lista pre-calculada de spawn points en orden DETERMINÍSTICO.
        all_spawn_points = self.map.get_spawn_points()
        random.seed(self.seed_val + 1000)
        random.shuffle(all_spawn_points)
        self.eval_spawn_points = all_spawn_points

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
        print(f"[EvalEnv] NPCs spawneados: {spawned}/{count}")
        for _ in range(10):
            self.world.tick()

    # ── Spawn ego y sensores ──

    def _spawn_vehicle(self):
        """Spawnea ego en el spawn point determinado por episode_count."""
        vehicle_bp = self.blueprint_library.filter("model3")[0]
        idx = self.episode_count % len(self.eval_spawn_points)
        for offset in range(20):
            sp = self.eval_spawn_points[(idx + offset) % len(self.eval_spawn_points)]
            vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                self.vehicle = vehicle
                self.actor_list.append(self.vehicle)
                return
        raise RuntimeError(f"No se pudo spawnear ego en episodio {self.episode_count}.")

    def _setup_camera(self):
        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IM_HEIGHT))
        cam_bp.set_attribute("fov", str(FOV))
        cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda img: self._process_image(img))

    def _setup_collision_sensor(self):
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _setup_lane_sensor(self):
        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))

    # ── Callbacks ──

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))
        self.front_camera = cv2.resize(array[:, :, :3], (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)

    def _on_collision(self, event):
        self.collision_flag = True

    def _on_lane_invasion(self, event):
        self.lane_invasion_flag = True
        self._lane_invasions += 1

    # ── Utilidades ──

    def _get_speed_kmh(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def _is_wrong_way(self, wp_fwd):
        vel = self.vehicle.get_velocity()
        speed_vec = math.sqrt(vel.x**2 + vel.y**2)
        if speed_vec < 0.5:
            return False, 0.0
        vel_dot = (vel.x / speed_vec) * wp_fwd.x + (vel.y / speed_vec) * wp_fwd.y
        return vel_dot < 0.0, vel_dot

    def _get_distance_traveled(self):
        loc = self.vehicle.get_transform().location
        if self.prev_location is None:
            self.prev_location = loc
            return 0.0
        dist = math.sqrt((loc.x - self.prev_location.x)**2 + (loc.y - self.prev_location.y)**2)
        self.prev_location = loc
        return dist

    # ── Observación ──

    def _get_vector_obs(self):
        """Vector de 10 mediciones — DEBE ser idéntico al training."""
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
        img = self.front_camera.copy() if self.front_camera is not None else np.zeros((OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), dtype=np.uint8)
        return {"image": img, "vector": self._get_vector_obs()}

    # ── Telemetría ──

    def _record_telemetry(self, action):
        """Guarda una fila de telemetría por step para análisis posterior."""
        vt = self.vehicle.get_transform()
        wp = self.map.get_waypoint(vt.location, project_to_road=True)
        wp_fwd = wp.transform.get_forward_vector()
        veh_fwd = vt.get_forward_vector()

        speed = self._get_speed_kmh()
        is_wrong, vel_dot = self._is_wrong_way(wp_fwd)
        orientation = veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y
        dist_to_center = vt.location.distance(wp.transform.location)

        steer, throttle, brake = DISCRETE_ACTIONS[action]

        self.episode_telemetry.append({
            'step': self.step_count,
            'pos_x': vt.location.x,
            'pos_y': vt.location.y,
            'speed_kmh': speed,
            'orientation': orientation,
            'dist_to_center': dist_to_center,
            'vel_dot': vel_dot,
            'is_wrong_way': int(is_wrong),
            'action': action,
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'on_driving_lane': int(wp.lane_type == carla.LaneType.Driving),
        })

    # ── Interfaz Gymnasium ──

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Limpieza segura de actores del episodio anterior:
        # parar sensores primero (evita callbacks zombie), después destruir.
        for actor in self.actor_list:
            try:
                if actor is not None and hasattr(actor, 'stop'):
                    actor.stop()
            except Exception:
                pass
        for actor in reversed(self.actor_list):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.actor_list = []

        self.collision_flag = False
        self.lane_invasion_flag = False
        self.front_camera = None
        self.step_count = 0
        self.stall_counter = 0
        self.prev_steer = 0.0
        self.prev_location = None
        self.episode_telemetry = []
        self._lane_invasions = 0
        self.episode_count += 1

        self._spawn_vehicle()
        self._setup_camera()
        self._setup_collision_sensor()
        self._setup_lane_sensor()

        # Esperar a que la cámara entregue el primer frame.
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
        """Ejecuta acción. Devuelve obs, reward=0 (no se usa), terminated, truncated, info."""
        self.step_count += 1

        steer, throttle, brake = DISCRETE_ACTIONS[action]
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self.world.tick()

        self._record_telemetry(action)

        speed = self._get_speed_kmh()
        terminated = False
        termination_cause = None

        # Stall.
        if speed < 2.0:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        if self.stall_counter >= STALL_TERMINATE_STEPS:
            terminated = True
            termination_cause = "stall"

        # Colisión.
        if self.collision_flag:
            terminated = True
            termination_cause = "collision"

        # Offroad.
        vt = self.vehicle.get_transform()
        wp = self.map.get_waypoint(vt.location, project_to_road=True)
        if wp.lane_type != carla.LaneType.Driving:
            terminated = True
            termination_cause = "offroad"

        self.lane_invasion_flag = False
        self.prev_steer = steer

        truncated = self.step_count >= MAX_STEPS
        if truncated and not terminated:
            termination_cause = "success"

        obs = self._build_obs()
        info = {
            "speed_kmh": speed,
            "termination_cause": termination_cause,
            "lane_invasions": self._lane_invasions,
        }
        return obs, 0.0, terminated, truncated, info

    def close(self):
        """
        Limpieza robusta. CARLA tira RuntimeError si destruimos un actor que ya
        no existe (callback en vuelo, NPC borrado por traffic manager, etc.).
        Cada destrucción va en su propio try/except para que un fallo no rompa
        la cadena. También detenemos los listeners de sensores ANTES de destruir.
        """
        # 1. Detener listeners de sensores.
        for actor in self.actor_list:
            try:
                if actor is not None and hasattr(actor, 'stop'):
                    actor.stop()
            except Exception:
                pass

        # 2. Destruir actores ego.
        for actor in reversed(self.actor_list):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.actor_list = []

        # 3. Destruir NPCs.
        for npc in self.npc_vehicles:
            try:
                if npc is not None and npc.is_alive:
                    npc.destroy()
            except Exception:
                pass
        self.npc_vehicles = []

        # 4. Restaurar settings.
        try: self.traffic_manager.set_synchronous_mode(False)
        except Exception: pass
        try: self.world.apply_settings(self._original_settings)
        except Exception: pass