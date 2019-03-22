from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt

env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=False, realtime_mode=False)
obs = env.reset()
plt.imshow(obs[0])
plt.show()
