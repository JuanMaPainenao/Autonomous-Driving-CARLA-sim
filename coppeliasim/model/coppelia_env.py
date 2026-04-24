import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class CoppeliaEnv(gym.Env):

 
    SENSOR_MAX_RANGE = 1.0
    SENSOR_INDICES = [2, 3, 4, 5]  #los 4 sensores frontales
    def __init__(self):
        super().__init__()

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

        observation = None
        info={}
        return observation, info

    def step(self, action):
        observation = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
