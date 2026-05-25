"""
Entorno Gymnasium para CARLA — Modelo 3'_A (M3 prime A).
Refinamiento cuantitativo de M3: collision penalty amplificada de -10 a -50.

Hipótesis: el problema de M3 es solo de balance numérico. El agente acumula
~1.5 de reward por step durante cientos de steps, lo cual hace que -10 de
penalty por colisión no sea suficiente disuasivo. Aumentando 5x el penalty,
la colisión debería volverse incosteable.

Todo lo demás (gate, observación, acciones, hiperparámetros) IDÉNTICO a M3.
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

IM_WIDTH, IM_HEIGHT, IM_CHANNELS, FOV = 640, 480, 3, 110
OBS_WIDTH, OBS_HEIGHT = 160, 120
VECTOR_OBS_SIZE = 10

FIXED_DELTA = 0.05
MAX_STEPS = 2000
TARGET_SPEED_KMH = 30

R_SPEED_MAX = 1.0
R_PROGRESS_PER_METER = 1.0
R_LANE_CENTER_MAX = 0.5
R_WRONG_WAY_PENALTY = -2.0

GATE_LANE_TOLERANCE = 1.5
GATE_SPEED_TOLERANCE = 25.0
GATE_ORIENT_SHARPNESS = 1.0

R_LANE_PENALTY = -5.0
R_COLLISION_PENALTY = -50.0    # ← ÚNICO CAMBIO vs M3: era -10.0
R_OFFROAD_PENALTY = -5.0

R_STALL_PENALTY = -5.0
STALL_TERMINATE_STEPS = 30

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
        self.log_path = f"logs/reward_log_M3A_{ts}.csv"
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
        print(f"[CarlaEnv] NPCs spawneados: {spawned}/{count}")
        for _ in range(10):
            self.world.tick()

    def _spawn_vehicle(self):
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

    def _get_vector_obs(self):
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

    def _compute_safety_gate(self, speed, vel_dot, dist_to_center, orientation, is_wrong):
        """Idéntico a M3."""
        g_orient = math.pow(max(0.0, (orientation + 1.0) / 2.0), GATE_ORIENT_SHARPNESS)
        g_lane = max(0.0, 1.0 - dist_to_center / GATE_LANE_TOLERANCE)
        g_speed = math.exp(-((speed - TARGET_SPEED_KMH) / GATE_SPEED_TOLERANCE) ** 2)
        g_direction = 0.0 if is_wrong else 1.0
        gate = g_orient * g_lane * g_speed * g_direction
        return gate, g_orient, g_lane, g_speed, g_direction

    def _compute_reward(self, action):
        """
        M3'_A — Idéntico a M3 pero con R_COLLISION_PENALTY = -50.0 (vs -10.0).
        """
        speed = self._get_speed_kmh()
        terminated = False

        vt = self.vehicle.get_transform()
        wp = self.map.get_waypoint(vt.location, project_to_road=True)
        wp_fwd = wp.transform.get_forward_vector()
        veh_fwd = vt.get_forward_vector()

        is_wrong, vel_dot = self._is_wrong_way(wp_fwd)
        vel_dot = float(np.clip(vel_dot, -1.0, 1.0))
        orientation = veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y
        dist_to_center = vt.location.distance(wp.transform.location)

        gate, g_orient, g_lane_factor, g_speed, g_direction = self._compute_safety_gate(
            speed, vel_dot, dist_to_center, orientation, is_wrong
        )

        r_speed = R_SPEED_MAX * math.exp(-((speed - TARGET_SPEED_KMH) / 20.0) ** 2)
        dist = self._get_distance_traveled()
        r_progress_raw = R_PROGRESS_PER_METER * dist
        r_lane_center_raw = R_LANE_CENTER_MAX * max(0.0, 1.0 - dist_to_center / 1.5)

        r_progress = gate * r_progress_raw
        r_lane_center = gate * r_lane_center_raw

        r_wrong_way = R_WRONG_WAY_PENALTY if is_wrong else 0.0

        steer, _, _ = DISCRETE_ACTIONS[action]
        self.prev_steer = steer

        if speed < 2.0:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        r_stall = 0.0
        if self.stall_counter >= STALL_TERMINATE_STEPS:
            r_stall = R_STALL_PENALTY
            terminated = True

        r_lane = R_LANE_PENALTY if self.lane_invasion_flag else 0.0
        self.lane_invasion_flag = False

        r_collision = 0.0
        if self.collision_flag:
            r_collision = R_COLLISION_PENALTY   # ← AHORA ES -50
            terminated = True

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

    def _log_rewards(self, components, total):
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

    def reset(self, seed=None, options=None):
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

    def _render_preview(self, obs_image, action, reward, components):
        display = cv2.resize(obs_image, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_NEAREST)
        speed = self._get_speed_kmh()
        names = {0:"IZQ+Th",1:"RECTO+Th",2:"DER+Th",3:"IZQ+Full",4:"RECTO+Full",5:"DER+Full",6:"IZQ+Freno",7:"RECTO+Freno",8:"DER+Freno"}
        cv2.putText(display, f"Speed:{speed:.0f} | {names.get(action,'?')} | Step:{self.step_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(display, f"R:{reward:+.2f} | Gate:{components['gate']:.2f} | Ep:{self._ep_reward:+.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("CARLA Agent View - M3'_A", display)
        cv2.waitKey(1)

    def close(self):
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