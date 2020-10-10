
import numpy as np
from envs.gym import GymEnvs

import moviepy.editor as mpy

env = GymEnvs('Ant-v3')
print(env)

act_shape = env.action_space.shape
env.reset()
movie_images = [env.render()]
for i in range(100):
    action = np.random.rand(*env.action_space.shape)*2-1
    o, r, d, info = env.step(action)
    movie_images.append(env.render())

clip = mpy.ImageSequenceClip(movie_images, fps=5)
clip.write_videofile('test.mp4')
