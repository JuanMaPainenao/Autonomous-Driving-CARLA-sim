from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

sim.loadScene('/opt/coppeliaSim/scenes/ParedLineas.ttt')

for i in range(200):
    try:
        alias = sim.getObjectAlias(i, 1)
        print(f'Handle {i}: {alias}')
    except:
        pass