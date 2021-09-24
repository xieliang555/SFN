import gym
import gymEnv
print(gymEnv.__file__)

import pybullet as p
import time
import pybullet_data
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from utils.vec_env.subproc_vec_env import SubprocVecEnv
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import cv2
import glob
import os
import pybullet as p 
# os.environ['PYOPENGL_PLATFORM'] = 'egl'



env = gym.make('gymEnv:peg-in-hole-v11', peg_type='square-fillet1', seed=1, test_mode=True)
obs = env.reset() 


for i in range(100):
	plt.subplot(1,2,1)
	plt.imshow(obs['img'].transpose((1,2,0)))
	plt.subplot(1,2,2)
	plt.imshow(obs['gt'])
	plt.show()
	obs, reward, done,_ = env.step(5)
	print('r', reward, 'done', done)

exit(0)

for i in range(10):
	# print(obs['anchor'].shape)
	# print(obs['pos'].shape)
	# print(obs['neg'].shape)
	# plt.imshow(obs['img_peg'].transpose((1,2,0)))
	# plt.show()

	obs,_,_,_ = env.step(0)
exit(0)

 


# p.connect(p.GUI)
# # p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0,0,-9.8)
# # pandaId = p.loadURDF("franka_panda/panda.urdf", np.array([0,0,0]), useFixedBase=True)
# # p.enableJointForceTorqueSensor(pandaId, 8, 1)
# hole_id = p.loadURDF('./gymEnv/envs/mesh/square-e/hole/hole.urdf', 
# 	basePosition=[0,0,0], globalScaling=0.05, useFixedBase=1,
# 	baseOrientation=p.getQuaternionFromEuler([math.pi/2,0,0]))
# peg_id = p.loadURDF('./gymEnv/envs/mesh/square-e/peg/peg_test.urdf', 
# 	basePosition=[0,0,0.005], globalScaling=0.015, useFixedBase=0,
# 	baseOrientation=p.getQuaternionFromEuler([math.pi/2,0,0]))


# while 1:
# 	time.sleep(0.5)
# 	# print(p.getJointState(pandaId, 8)[2])
# 	p.stepSimulation()

p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.resetDebugVisualizerCamera( cameraDistance=1, cameraYaw=0,
cameraPitch=-45, cameraTargetPosition=[0,0,0])

peg = p.loadURDF('./gymEnv/envs/mesh/test/peg.urdf', 
	basePosition=[0,0,-0.001], 
	baseOrientation = p.getQuaternionFromEuler([math.pi/2, 0, 0]),
	useFixedBase=0, globalScaling=1)
hole = p.loadURDF('./gymEnv/envs/mesh/test/base.urdf', 
	basePosition=[0,0,0], 
	baseOrientation = p.getQuaternionFromEuler([math.pi/2, 0, 0]),
	useFixedBase=1, globalScaling=1)

time.sleep(2)
while 1:
	time.sleep(1/240.)
	p.stepSimulation()
exit(0)








if __name__ == '__main__':
	envs = []
	seed = 7
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v3', peg_type='square-triangle', seed=seed))
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v3', peg_type='square-square', seed=seed+1))
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v3', peg_type='square-pentagon', seed=seed+2))
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v3', peg_type='square-hexagon', seed=seed+3))
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v3', peg_type='square-trapezoid', seed=seed+4))
	venv = SubprocVecEnv(envs)
	obs = venv.reset()

	# done = False

	print(obs['img'].shape)
	print(obs['gt'].shape)

	for i in range(100):
		plt.subplot(2,5,1)
		plt.imshow(obs['img'][0].transpose((1,2,0)))
		plt.subplot(2,5,2)
		plt.imshow(obs['img'][1].transpose((1,2,0)))
		plt.subplot(2,5,3)
		plt.imshow(obs['img'][2].transpose((1,2,0)))
		plt.subplot(2,5,4)
		plt.imshow(obs['img'][3].transpose((1,2,0)))
		plt.subplot(2,5,5)
		plt.imshow(obs['img'][4].transpose((1,2,0)))

		plt.subplot(2,5,6)
		plt.imshow(obs['gt'][0])
		plt.subplot(2,5,7)
		plt.imshow(obs['gt'][1])
		plt.subplot(2,5,8)
		plt.imshow(obs['gt'][2])
		plt.subplot(2,5,9)
		plt.imshow(obs['gt'][3])
		plt.subplot(2,5,10)
		plt.imshow(obs['gt'][4])
		plt.show()


		a = [0,0,0,0,0]
		obs, r, d, info = venv.step(a)
		print(d)


	venv.close()