from networkx.algorithms.smallworld import sigma
from numpy.random.mtrand import rand
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import numpy as np 
from PIL import Image
from torch.utils import data

from utils.utils import get_position_gt
from utils.vec_env.subproc_vec_env import SubprocVecEnv
from utils.pck_acc import accuracy

import gym
import gymEnv

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

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['CUDA_VISIBLE_DEVICES']= "1"

np.random.seed(1)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-triangle', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-square', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-pentagon', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-hexagon', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-diamond', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-trapezoid', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-fillet1', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-fillet2', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-fillet3', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-fillet4', seed=1, test_mode=False)
env = gym.make('gymEnv:peg-in-hole-v11', peg_type='square-concave1', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-convex1', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-convex2', seed=1, test_mode=False)
# env = gym.make('gymEnv:peg-in-hole-v12', peg_type='square-convex3', seed=1, test_mode=False)


# input: 分割图像 {'gt':gt, 'dxy':dxy, 'dyaw':dyaw}
def get_dyaw(o, device):
	with torch.no_grad():
		seg_pred = o['gt']
		peg_pred = np.float32(seg_pred==1)
		hole_pred = np.float32(seg_pred==2)
		
		rotation_ls = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
		idx_pos = abs(o['dyaw']-rotation_ls).argmin()
		theta_pos = rotation_ls[idx_pos]
		# print('dyaw/gt dyaw', o['dyaw'],'/',theta_pos)

		hole_rots = np.array([np.array(Image.fromarray(np.uint8(hole_pred)).rotate(t, 
			expand=False, fillcolor=0)) for t in rotation_ls])
		
		# print(hole_rots.shape)
		# peg_pred = np.uint8(seg_pred==1)
		# for i in range(11):
		# 	plt.subplot(3,4,i+1)
		# 	plt.imshow(cv2.addWeighted(np.array(hole_rots[i]),1,peg_pred,2,0))

		inputs = torch.cat((
			torch.as_tensor(peg_pred, dtype=torch.float32).unsqueeze(0),
			torch.as_tensor(hole_rots, dtype=torch.float32)), 0)
		# print(inputs.shape)
		outs = pose_model(inputs.unsqueeze(1).to(device))
		# print(outs.shape)

		anchor_flatten = outs[0].view(1,-1) 	# torch.Size([1, 150000])
		match_flatten = outs[1:].view(11,-1) 	# torch.Size([11, 150000])

		# for i in range(11):
		# 	plt.subplot(3,4,i+1)
		# 	plt.imshow(outs[1:][i].permute(1,2,0))
		# plt.show()
		
		dis = F.pairwise_distance(anchor_flatten.tile((11,1)), match_flatten)
		theta = rotation_ls[dis.argmin()]
		# print('pred theta', theta)
		
	return theta

# input: 分割图像 {'img':img, 'gt':gt, 'dxy':dxy, 'dyaw':dyaw}
def get_dxy(o, device):
	with torch.no_grad():
		seg_pred = torch.as_tensor(o['gt'], dtype=torch.float32)
		outs = position_model(seg_pred.unsqueeze(0).unsqueeze(0).to(device))
		preds = (outs[:,1,:,:].max() == outs[:,1,:,:]).type(torch.long)
		# plt.subplot(1,2,1)
		# plt.imshow(o['gt'])
		# plt.subplot(1,2,2)
		# plt.imshow(preds.permute(1,2,0))
		# plt.show()
		n,h,w = preds.size()
		preds_flatten = preds.view(n,-1)
		idx = np.argmax(preds_flatten.detach().cpu().numpy(), axis=1)
		dy, dx = idx // w, idx % w
		# 图像坐标系转世界坐标系
		dx = -(dx-10)/1000
		dy = (dy-10)/1000
		print(dx,dy)

		# position_gts = get_position_gt(o)
		# # plt.plot(position_gts.transpose(1,2,0))
		# # position_gt = position_gts.transpose(1,2,0)
		# # print(position_gt.shape)
		# # plt.imshow(position_gt[:,:,0])
		# # plt.show()
		# # exit()

		# _, acc_position_5, _, _ = accuracy(
		# 	preds.unsqueeze(1).detach().cpu().numpy(),
		# 	np.expand_dims(position_gts, axis=1), thr=5)
		# print('acc_position_5', acc_position_5, 'pred_dxy', dx, dy, 'true_dxy', o['dxy'][0], o['dxy'][1])
	return dx, dy, dy, dx

	
def main(iterates=200, steps = 10):
	success_dyaw = 0
	success_dxy = 0
	success_insert = 0

	for ite in range(iterates):    
		print(ite)
		obs = env.reset()
        # plt.imshow(obs['img'].transpose(1,2,0))
		# plt.show()
		# plt.imshow(obs['gt'])
		# plt.show()  
        
        ############### one-step ###############
	# 	print('true dyaw', obs['dyaw'])
	# 	dyaw = get_dyaw(obs,args.device)
	# 	print('pred_dyaw', dyaw)       

		print('true dx', obs['dxy'][0], 'true_dy', obs['dxy'][1])
		dx, dy, _, _ = get_dxy(obs,args.device)
		print('pred_dx', dx, 'pred_dy', dy)

		# if abs(dyaw - obs['dyaw']) <= 2:
			# success_dyaw += 1
		if abs(dx[0] -  obs['dxy'][0]) <= 0.001 and abs(dy[0] -  obs['dxy'][1]) <= 0.001:
			success_dxy += 1

	# print('success_dyaw', success_dyaw, '/', iterates)
	print('success_dxy', success_dxy, '/', iterates)

        ############### close-loop orientation###############
	# 	dyaw_done = False
	# 	for _ in range(steps):
	# 		dyaw = get_dyaw(obs,args.device)
	# 		obs, _, _, _ = env.step([0, 0, dyaw])
	# 		if abs(dyaw - obs['dyaw']) <= 2:
	# 			dyaw_done = True
	# 			break
	# 	if dyaw_done:
	# 		success_dyaw += 1
	# print('success_dyaw', success_dyaw, '/', iterates, success_dyaw / iterates)
        ############### close-loop orientation###############

       ############### close-loop position###############
	# 	dxy_done = False
	# 	for _ in range(steps):
	# 		print('true dx', obs['dxy'][0], 'true_dy', obs['dxy'][1])
	# 		dx, dy, _, _ = get_dxy(obs,args.device)
	# 		obs, _, _, _ = env.step([dx, dy, 0])
	# 		if abs(dx - obs['dxy'][0]) <= 0.001 and abs(dy - obs['dxy'][1]) <= 0.001:
	# 			dxy_done = True
	# 			break
	# 	if dxy_done:
	# 		success_dxy += 1
	# print('success_dxy', success_dxy, '/', iterates, success_dxy / iterates)
        ############## close-loop position###############


       ############### close-loop insertion###############
	# 	done = False
	# 	for _ in range(steps):
	# 		dyaw = get_dyaw(obs,args.device)
	# 		dx, dy, _, _ = get_dxy(obs,args.device)
	# 		obs, _, _, _ = env.step([dx, dy, dyaw])
	# 		if abs(dyaw - obs['dyaw']) <= 2 and abs(dx - obs['dxy'][0]) <= 0.001 and abs(dy - obs['dxy'][1]) <= 0.001:
	# 			done = True
	# 			break
	# 	if done:
	# 		success_insert += 1    
	# print('success_insert', success_insert, '/', iterates, success_insert / iterates)
        ############### close-loop insertion###############


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--position_model_path', type=str, default='/home/xieliang/桌面/position_11_ff_resume/position_11_ff_resume_s1/pyt_save/model.pt')
	parser.add_argument('--pose_model_path', type=str, default='')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--iterates', type=int, default=200)
	parser.add_argument('--steps', type=int, default=10)
	args = parser.parse_args()

	# pose_model = torch.load(args.pose_model_path,map_location=args.device)
	position_model = torch.load(args.position_model_path,map_location=args.device)
	
	main(args.iterates, args.steps)
