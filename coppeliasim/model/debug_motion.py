# debug_motion.py
from coppelia_env import CoppeliaEnv

env = CoppeliaEnv(reward_mode='R1')
obs, _ = env.reset()

print("\n=== TEST DE MOVIMIENTO ===")
print(f"Posición inicial: {env.sim.getObjectPosition(env.robot, -1)}")
print(f"Observación inicial: {obs}")

print("\nForzando acción 0 (avanzar) durante 50 steps...\n")
for i in range(50):
    obs, reward, term, trunc, info = env.step(0)  # SIEMPRE avanzar
    
    if i % 5 == 0:
        pos = env.sim.getObjectPosition(env.robot, -1)
        # Leer velocidad real del joint
        left_vel  = env.sim.getJointVelocity(env.left_motor)
        right_vel = env.sim.getJointVelocity(env.right_motor)
        # Velocidad lineal del robot
        lin_vel, _ = env.sim.getObjectVelocity(env.robot)
        speed = (lin_vel[0]**2 + lin_vel[1]**2)**0.5
        
        print(f"Step {i:2d}: "
              f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}) | "
              f"joint_vel=(L={left_vel:+.2f}, R={right_vel:+.2f}) | "
              f"speed={speed:.3f} m/s")
    
    if term:
        print(f">>> Terminó por colisión en step {i}")
        break
    if trunc:
        print(f">>> Truncado en step {i}")
        break

env.close()
print("\n=== FIN TEST ===")