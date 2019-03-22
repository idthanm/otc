from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
im = Image.open('1.jpg')

# a = np.random.random((84, 84, 3))
im_arr = np.array(im)
b = im.convert('L')
b_ = np.array(b)

plt.imshow(a)
plt.show()