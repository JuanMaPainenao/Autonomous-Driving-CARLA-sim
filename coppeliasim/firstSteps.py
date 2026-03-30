from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')
sim.stopSimulation()

sim.loadScene('/opt/coppeliaSim/scenes/ParedLineas.ttt')
sim.startSimulation()

rightMotor = sim.getObject('/PioneerP3DX/rightMotor')
leftMotor = sim.getObject('/PioneerP3DX/leftMotor')

print('Right motor handle: ', rightMotor)
print('Left motor handle: ', leftMotor)

# sim.getObject() obtiene el handle del script interno del robot
script = sim.getObject('/PioneerP3DX/Script')

# sim.removeObject() elimina un objeto de la escena (en este caso el script)
sim.removeObject(script)

sim.setJointTargetVelocity(rightMotor, -2.0)
sim.setJointTargetVelocity(leftMotor, 2.0)