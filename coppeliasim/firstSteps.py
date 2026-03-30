from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math

client = RemoteAPIClient()
sim = client.require('sim')
sim.stopSimulation()
time.sleep(0.5)

#sim.loadScene('/opt/coppeliaSim/scenes/ParedLineas.ttt')
sim.startSimulation()

rightMotor = sim.getObject('/PioneerP3DX/rightMotor')
leftMotor = sim.getObject('/PioneerP3DX/leftMotor')
robot = sim.getObject('/PioneerP3DX')

print('Right motor handle: ', rightMotor)
print('Left motor handle: ', leftMotor)

#script = sim.getObject('/PioneerP3DX/Script')
#sim.removeObject(script)

def steer_right(sim, rightMotor, leftMotor, robot_handle):
    orientacion_inicial = sim.getObjectOrientation(robot_handle, -1)
    yaw_inicial = orientacion_inicial[2]
    yaw_objetivo = yaw_inicial - math.pi / 2

    while True:
        orientacion_actual = sim.getObjectOrientation(robot_handle, -1)
        yaw_actual = orientacion_actual[2]
        diff = yaw_actual - yaw_objetivo
        diff = math.atan2(math.sin(diff), math.cos(diff))

        if abs(diff) < 0.02:
            break
        vel = min(1.5, max(0.1, abs(diff) * 2.0))

        sim.setJointTargetVelocity(rightMotor, -vel)
        sim.setJointTargetVelocity(leftMotor, vel)
        time.sleep(0.01)

    sim.setJointTargetVelocity(rightMotor, 0.0)
    sim.setJointTargetVelocity(leftMotor, 0.0)

while(1):
    sim.setJointTargetVelocity(rightMotor, 2.0)
    sim.setJointTargetVelocity(leftMotor, 2.0)
    time.sleep(2)
    steer_right(sim, rightMotor, leftMotor, robot)

