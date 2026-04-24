from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math

# Conexión y setup
client = RemoteAPIClient()
sim = client.require('sim')
sim.stopSimulation()
time.sleep(0.5)

sim.setStepping(True)  # Modo sincrónico: la simulación avanza solo cuando llamamos client.step()
sim.startSimulation()

# Handles
rightMotor = sim.getObject('/PioneerP3DX/rightMotor')
leftMotor  = sim.getObject('/PioneerP3DX/leftMotor')
robot      = sim.getObject('/PioneerP3DX')
sensor3    = sim.getObject('/PioneerP3DX/ultrasonicSensor[3]')
sensor4    = sim.getObject('/PioneerP3DX/ultrasonicSensor[4]')

print('Right motor handle:', rightMotor)
print('Left motor handle: ', leftMotor)
print('Sensor [3] handle: ', sensor3)
print('Sensor [4] handle: ', sensor4)


# Lectura de sensores
def read_sensor(sensor_handle):
    """Devuelve la distancia detectada en metros, o None si no detecta nada."""
    result, distance, point, obj, normal = sim.readProximitySensor(sensor_handle)
    if result == 1:
        return distance
    return None


def print_front_sensors():
    """Imprime las lecturas de los sensores frontales centrales."""
    d3 = read_sensor(sensor3)
    d4 = read_sensor(sensor4)
    s3 = f'{d3:.3f} m' if d3 is not None else 'sin detección'
    s4 = f'{d4:.3f} m' if d4 is not None else 'sin detección'
    print(f'  Sensor[3]: {s3:>15}  |  Sensor[4]: {s4:>15}')


# Navegación
def steer_right(sim, client, rightMotor, leftMotor, robot_handle):
    """Gira 90° a la derecha en el lugar."""
    orientacion_inicial = sim.getObjectOrientation(robot_handle, -1)
    yaw_inicial  = orientacion_inicial[2]
    yaw_objetivo = yaw_inicial - math.pi / 2

    while True:
        client.step()
        print_front_sensors()

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
        print('\n>>> Avanzando...')
        for _ in range(160):
            client.step()
            print_front_sensors()

        # Girar a la derecha
        print('\n>>> Girando a la derecha...')
        steer_right(sim, client, rightMotor, leftMotor, robot)

except KeyboardInterrupt:
    print('\nDeteniendo...')

finally:
    sim.setJointTargetVelocity(rightMotor, 0.0)
    sim.setJointTargetVelocity(leftMotor,  0.0)
    sim.stopSimulation()
    print('Simulación detenida.')