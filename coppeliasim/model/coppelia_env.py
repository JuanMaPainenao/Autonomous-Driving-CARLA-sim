import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class CoppeliaEnv(gym.Env):

 
    SENSOR_MAX_RANGE = 1.0
    SENSOR_INDICES = [2, 3, 4, 5]  #los 4 sensores frontales
    MAX_EPISODE_STEPS = 500
    COLLISION_THRESHOLD = 0.1



    def __init__(self, reward_mode = 'R1'):
        super().__init__()


        assert reward_mode in ['R1','R2','R3'], \
            f"Modo de recompensa inválido: {reward_mode}. Debe ser R1, R2 o R3."
        self.reward_mode = reward_mode

        self.client = RepoteAPIClient()
        self.sim = self.client.require('sim')


        self.sim.stopSimulation()

        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            pass
        
        self.sim.setStepping(True)

        self.robot = self.sim.getObject('/PioneerP3DX')
        self.left_motor = self.sim.getObject('/PioneerP3DX/leftMotor')
        self.right_motor = self.sim.getObject('/PioneerP3DX/rightMotor')

        self.sensors = []
        for idx in self.SENSOR_INDICES:
            handle = self.sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{idx}]')
            self.sensors.append(handle)

        self.initial_position = self.sim.getObjectPosition(self.robot, -1)
        self.initial_orientation = self.sim.getObjectOrientation(self.robot, -1)


        # Definir espacios
        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape=(6,),
            dtype = np.float32
        )

        self.action_space = spaces.Discrete(3)
        self.prev_action = 0


        print(f'CoppeliaEnv conectado. Robot handle: {self.robot}')
        print(f'CoppeliaEnv {len(self.sensors)} sensores cargados: {self.SENSOR_INDICES}')
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            pass
        
        self.sim.setObjectPosition(self.robot, -1, self.initial_position)
        self.sim.setObjectOrientation(self.robot, -1, self.initial_orientation)

        self.prev_action = 0
        self.step_count = 0

        self.sim.startSimulation()

        self.client.step()

        observation = self._get_observation()
        info = {}

    def _get_observation(self):
        sensor_readings = []
        for sensor_handle in self.sensors:
            result, distance, _, _, _ = self.sim.readProximitySensor(sensor_handle)
            if result == 1:
                norm_dist = distance / self.SENSOR_MAX_RANGE
            else:
                norm_dist = 1.0
                
            sensor_readings.append(norm_dist)
        linear_vel, _ = self.sim.getObjectVelocity(self.robot)

        speed = np.sqrt(linear_vel[0]**2 + linear_vel[1]**2)
        MAX_SPEED = 1.0
        speed_norm = min(speed / MAX_SPEED, 1.0)

        prev_action_norm = self.prev_action / 2.0

        obs = np.array(
            sensor_readings + [speed_norm, prev_action_norm],
            dtype = np.float32
        )

        return obs

    
        return observation, info

    def step(self, action):
        left_vel, right_vel = self._action_to_motor_velocities(action)

        self.sim.setJointTargetVelocity(self.left_motor, left_vel)
        self.sim.setJointTargetVelocity(self.right_motor, right_vel)

        self.client.step()

        observation = self._get_observation()
        reward, reward_components = self._compute_reward(observation)

        terminated = False

        self.step_count += 1
        truncated = self.step_count >= self.MAX_EPISODE_STEPS

        self.prev_action = action

        info = dict(reward_components)
        return observation, reward, terminated, truncated, info

    def _action_to_motor_velocities(self, action):
        if action ==0:
            return 2.0, 2.0
        elif action == 1:
            return 0.5, 2.0
        elif action == 2:
            return 2.0, 0.5
        else:
            raise ValueError(f'Acción inválida: {action}')


    def _compute_reward(self, observation):
        s2, s3, s4, s5, v_norm, _ = observation
        d_min = min(s2, s3, s4, s5)
        collision = d_min < self.COLLISION_THRESHOLD

        if self.reward_mode == 'R1':
            return self._reward_r1(v_norm, collision)
        elif self.reward_mode == 'R2':
            return self._reward_r2(v_norm, collision, d_min)
        elif self.reward_mode == 'R3':
            return self._reward_r3(v_norm, collision, d_min)

    def _reward_r1(self, v_norm, collision):
        K_V = 1.0
        K_COL = 100.0

        r_velocity = K_V * v_norm
        r_collision = -K_COL if collision else 0.0

        reward = r_velocity + r_collision
        components = {
            'r_velocity': r_velocity,
            'r_collision': r_collision
        }
        return reward, components

    def _reward_r2(self, v_norm, collision, d_min):
        K_V = 1.0
        K_COL = 100.0
        K_PROX = 5.0
        D_SAFE = 0.3

        r_velocity = K_V * v_norm
        r_collision = -K_COL if collision else 0.0
        r_proximity = -K_PROX * max(0.0, D_SAFE - d_min)

        reward = r_velocity + r_collision + r_proximity
        components = {
            'r_velocity': r_velocity,
            'r_collision': r_collision,
            'r_proximity': r_proximity
        }

        return reward, components

    def _reward_r3(self, v_norm, collision, d_min):
        K_V = 1.0
        K_COL = 100.0
        K_CLEAR = 2.0

        r_velocity = K_V * v_norm
        r_collision = -K_COL if collision else 0.0
        r_clearance = K_CLEAR * d_min

        reward = r_velocity + r_collision + r_clearance
        components = {
            'r_velocity': r_velocity,
            'r_collision': r_collision,
            'r_clearance': r_clearance
        }
        return reward, components


    def close(self):
        pass
