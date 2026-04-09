from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

client = RemoteAPIClient()
sim = client.require('sim')

# NO tocar la escena, abrila manualmente desde CoppeliaSim primero
sim.startSimulation()
time.sleep(0.5)

print("Objetos en la escena:")
for i in range(200):
    try:
        handle = sim.getObject('/', {'index': i})
        alias = sim.getObjectAlias(handle, 4)
        print(f"  [{i}] {alias} (handle: {handle})")
    except:
        print(f"--- Fin en índice {i} ---")
        break