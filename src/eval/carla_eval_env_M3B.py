"""
Entorno de evaluación para CARLA — versión M3'_B (observación de 11 valores).

Idéntico a carla_eval_env.py pero con la observación extendida a 11 valores
(agrega min_dist_to_npc normalizado), para que PPO.load() del modelo M3B
funcione sin mismatch de dimensiones.

NO computa reward — solo loggea telemetría para análisis.
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
    FIXED_DELTA, MAX_STEPS, TARGET_SPEED_KMH,
    STALL_TERMINATE_STEPS, DISCRETE_ACTIONS,
    NUM_NPC_VEHICLES, NPC_VEHICLE_MODELS, CLIENT_TIMEOUT,
)

# Vector de 11 valores para M3B (los 10 originales + min_dist_to_npc).
VECTOR_OBS_SIZE = 11

# Parámetros de proximidad (idénticos al env de entrenamiento de M3B).
PROXIMITY_DETECTION_RADIUS = 30.0
PROXIMITY_ANGLE_THRESHOLD = -0.3


class CarlaEvalEnv(gym.Env):
    """Entorno Gymnasium para evaluación de M3B. No computa reward."""

    metadata = {"render_modes": []}

    def __init__(self, town="Town10HD", with_npcs=True, seed=42):
        super().__init__()
        self.town = town
        self.with_npcs = with_npcs
        self.seed_val = seed

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), np.uint8),
            "vector": spaces.Box(-1.0, 2.0, (VECTOR_OBS_SIZE,), np.float32),
        })
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

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
        self.episode_telemetry = []

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(CLIENT_TIMEOUT)

        current_world = self.client.get_world()
        current_map_name = current_world.get_map().name
        if self.town in current_map_name:
            print(f"[EvalEnv-M3B] {self.town} ya está cargado (actual: {current_map_name}) — no se recarga.")
            self.world = current_world
        else:
            print(f"[EvalEnv-M3B] Cargando {self.town} (actual: {current_map_name})...")
            self.world = self.client.load_world(self.town)
            time.sleep(2.0)
            print(f"[EvalEnv-M3B] {self.town} cargado.")

        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.seed_val)

        random.seed(self.seed_val)

        if self.with_npcs:
            self._spawn_npc_traffic()

        all_spawn_points = self.map.get_spawn_points()
        random.seed(self.seed_val + 1000)
        random.shuffle(all_spawn_points)
        self.eval_spawn_points = all_spawn_points

    def _spawn_npc_traffic(self):
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
        print(f"[EvalEnv-M3B] NPCs spawneados: {spawned}/{count}")
        for _ in range(10):
            self.world.tick()

    def _spawn_vehicle(self):
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

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))
        self.front_camera = cv2.resize(array[:, :, :3], (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)

    def _on_collision(self, event):
        self.collision_flag = True

    def _on_lane_invasion(self, event):
        self.lane_invasion_flag = True
        self._lane_invasions += 1

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

    def _get_min_distance_to_npc(self):
        """
        Distancia al vehículo más cercano al frente/costado del ego.
        Idéntico al env de entrenamiento de M3B (mismo cálculo).
        """
        ego_loc = self.vehicle.get_location()
        ego_fwd = self.vehicle.get_transform().get_forward_vector()

        min_dist = PROXIMITY_DETECTION_RADIUS
        for npc in self.world.get_actors().filter('*vehicle*'):
            if npc.id == self.vehicle.id:
                continue
            npc_loc = npc.get_location()
            dist = ego_loc.distance(npc_loc)
            if dist > PROXIMITY_DETECTION_RADIUS:
                continue
            dx = npc_loc.x - ego_loc.x
            dy = npc_loc.y - ego_loc.y
            norm = math.sqrt(dx*dx + dy*dy)
            if norm < 1e-6:
                continue
            cos_angle = (ego_fwd.x * dx + ego_fwd.y * dy) / norm
            if cos_angle < PROXIMITY_ANGLE_THRESHOLD:
                continue
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_vector_obs(self):
        """Vector de 11 valores — IDÉNTICO al env de entrenamiento de M3B."""
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
        min_dist_npc = self._get_min_distance_to_npc()
        min_dist_npc_norm = min_dist_npc / PROXIMITY_DETECTION_RADIUS

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
            min_dist_npc_norm,   # 11vo valor
        ], dtype=np.float32)

    def _build_obs(self):
        img = self.front_camera.copy() if self.front_camera is not None else np.zeros((OBS_HEIGHT, OBS_WIDTH, IM_CHANNELS), dtype=np.uint8)
        return {"image": img, "vector": self._get_vector_obs()}

    def _record_telemetry(self, action):
        vt = self.vehicle.get_transform()
        wp = self.map.get_waypoint(vt.location, project_to_road=True)
        wp_fwd = wp.transform.get_forward_vector()
        veh_fwd = vt.get_forward_vector()

        speed = self._get_speed_kmh()
        is_wrong, vel_dot = self._is_wrong_way(wp_fwd)
        orientation = veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y
        dist_to_center = vt.location.distance(wp.transform.location)
        min_dist_npc = self._get_min_distance_to_npc()

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
            'min_dist_npc': min_dist_npc,   # columna extra para análisis de M3B
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        self.step_count += 1

        steer, throttle, brake = DISCRETE_ACTIONS[action]
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self.world.tick()

        self._record_telemetry(action)

        speed = self._get_speed_kmh()
        terminated = False
        termination_cause = None

        if speed < 2.0:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        if self.stall_counter >= STALL_TERMINATE_STEPS:
            terminated = True
            termination_cause = "stall"

        if self.collision_flag:
            terminated = True
            termination_cause = "collision"

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
        for npc in self.npc_vehicles:
            try:
                if npc is not None and npc.is_alive:
                    npc.destroy()
            except Exception:
                pass
        self.npc_vehicles = []
        try: self.traffic_manager.set_synchronous_mode(False)
        except Exception: pass
        try: self.world.apply_settings(self._original_settings)
        except Exception: pass
