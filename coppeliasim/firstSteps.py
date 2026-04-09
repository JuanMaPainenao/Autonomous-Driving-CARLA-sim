from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np
import cv2

# Conexión y setup
client = RemoteAPIClient()
sim = client.require('sim')
sim.stopSimulation()
time.sleep(0.5)

sim.setStepping(True)  # Modo sincrónico: la simulación avanza solo cuando llamamos client.step()
sim.startSimulation()

# Handles
rightMotor   = sim.getObject('/PioneerP3DX/rightMotor')
leftMotor    = sim.getObject('/PioneerP3DX/leftMotor')
robot        = sim.getObject('/PioneerP3DX')
visionSensor = sim.getObject('/PioneerP3DX/visionSensor')

print('Right motor handle:', rightMotor)
print('Left motor handle: ', leftMotor)
print('Vision sensor handle:', visionSensor)


# Cámara
def get_camera_frame(sim, client, sensor_handle):
    """Captura un frame del vision sensor y lo devuelve como array BGR para OpenCV."""
    client.step()  # Avanzar un paso de simulación para obtener datos frescos
    img, resolution = sim.getVisionSensorImg(sensor_handle)
    arr = np.frombuffer(img, dtype=np.uint8)
    arr = arr.reshape(resolution[1], resolution[0], 3)
    arr = np.flipud(arr)                        # CoppeliaSim devuelve la imagen invertida verticalmente
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # CoppeliaSim usa RGB, OpenCV usa BGR
    return bgr, resolution


# Navegación
def steer_right(sim, client, rightMotor, leftMotor, robot_handle, sensor_handle):
    """Gira 90° a la derecha en el lugar."""
    orientacion_inicial = sim.getObjectOrientation(robot_handle, -1)
    yaw_inicial  = orientacion_inicial[2]
    yaw_objetivo = yaw_inicial - math.pi / 2

    while True:
        client.step()

        # Mostrar cámara durante el giro
        img, resolution = sim.getVisionSensorImg(sensor_handle)
        arr = np.frombuffer(img, dtype=np.uint8)
        arr = arr.reshape(resolution[1], resolution[0], 3)
        arr = np.flipud(arr)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imshow('Vision Sensor', bgr)
        cv2.waitKey(1)

        orientacion_actual = sim.getObjectOrientation(robot_handle, -1)
        yaw_actual = orientacion_actual[2]
        diff = yaw_actual - yaw_objetivo
        diff = math.atan2(math.sin(diff), math.cos(diff))

        if abs(diff) < 0.02:
            break

        vel = min(1.5, max(0.1, abs(diff) * 2.0))
        sim.setJointTargetVelocity(rightMotor, -vel)
        sim.setJointTargetVelocity(leftMotor,   vel)

    sim.setJointTargetVelocity(rightMotor, 0.0)
    sim.setJointTargetVelocity(leftMotor,  0.0)


# Loop principal
try:
    while True:
        # Mover hacia adelante
        sim.setJointTargetVelocity(rightMotor, 2.0)
        sim.setJointTargetVelocity(leftMotor,  2.0)

        # Avanzar 40 steps (~2 segundos simulados a 50ms/step)
        for _ in range(40):
            frame, resolution = get_camera_frame(sim, client, visionSensor)
            cv2.imshow('Vision Sensor', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt

        # Girar a la derecha
        steer_right(sim, client, rightMotor, leftMotor, robot, visionSensor)

except KeyboardInterrupt:
    print('Deteniendo...')

finally:
    sim.setJointTargetVelocity(rightMotor, 0.0)
    sim.setJointTargetVelocity(leftMotor,  0.0)
    cv2.destroyAllWindows()
    sim.stopSimulation()
    print('Simulación detenida.')