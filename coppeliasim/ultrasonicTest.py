from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

# Conexión y setup
client = RemoteAPIClient()
sim = client.require('sim')
sim.stopSimulation()
time.sleep(0.5)

sim.setStepping(True)
sim.startSimulation()

# Handles
robot = sim.getObject('/PioneerP3DX')
rightMotor = sim.getObject('/PioneerP3DX/rightMotor')
leftMotor = sim.getObject('/PioneerP3DX/leftMotor')

# Obtener los 16 sensores ultrasónicos
sensors = []
for i in range(1, 17):
    handle = sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i-1}]')
    sensors.append(handle)

print(f'Se cargaron {len(sensors)} sensores ultrasónicos.')


def read_sensor(sensor_handle):
    """
    Lee un sensor de proximidad.
    Devuelve la distancia en metros, o None si no detecta nada.
    """
    result, distance, point, obj, normal = sim.readProximitySensor(sensor_handle)
    if result == 1:
        return distance
    return None


def print_readings(readings):
    """Imprime las lecturas de todos los sensores y destaca los de interés."""
    print('\n--- Lectura de sensores ---')
    for i, dist in enumerate(readings):
        if dist is not None:
            print(f'  Sensor [{i}]: {dist:.3f} m')
        else:
            print(f'  Sensor [{i}]: sin detección')

    print('\n>>> Sensores de interés:')
    for idx in [11, 12]:
        dist = readings[idx]
        if dist is not None:
            print(f'  ultrasonicSensor[{idx}]: {dist:.3f} m')
        else:
            print(f'  ultrasonicSensor[{idx}]: sin detección (fuera de rango)')


# Avance inicial de 5 segundos
print('\n>>> Avanzando 5 segundos...')
sim.setJointTargetVelocity(rightMotor, 2.0)
sim.setJointTargetVelocity(leftMotor, 2.0)

t_inicio = sim.getSimulationTime()
while sim.getSimulationTime() - t_inicio < 5.0:
    client.step()
    # Leer y mostrar sensores mientras avanza
    readings = [read_sensor(s) for s in sensors]
    print_readings(readings)

# Detener motores
sim.setJointTargetVelocity(rightMotor, 0.0)
sim.setJointTargetVelocity(leftMotor, 0.0)
print('\n>>> Avance terminado. Robot detenido.\n')


# Loop principal: lectura continua con robot detenido
try:
    while True:
        client.step()
        readings = [read_sensor(s) for s in sensors]
        print_readings(readings)
        time.sleep(0.2)

except KeyboardInterrupt:
    print('\nDeteniendo...')

finally:
    sim.setJointTargetVelocity(rightMotor, 0.0)
    sim.setJointTargetVelocity(leftMotor, 0.0)
    sim.stopSimulation()
    print('Simulación detenida.')