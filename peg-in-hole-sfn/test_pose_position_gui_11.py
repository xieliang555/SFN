import gym
import gymEnv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import numpy as np 
from PIL import Image
import time
import pybullet as p
import math

from utils.utils import get_position_gt
from utils.vec_env.subproc_vec_env import SubprocVecEnv
from utils.pck_acc import accuracy


np.random.seed(1)

def get_dyaw(o, device):
	with torch.no_grad():
		seg_pred = o['gt'][0]
		peg_pred = np.float32(seg_pred==1)
		hole_pred = np.float32(seg_pred==2)
		rotation_ls = np.array([-10,-8,-6,-4,-2, 0, 2,4,6,8,10])
		idx_pos = abs(o['dyaw']-rotation_ls).argmin()
		theta_pos = rotation_ls[idx_pos]
		print('dyaw/gt dyaw', o['dyaw'],'/',theta_pos)

		hole_rots = np.array([np.array(Image.fromarray(np.uint8(hole_pred)).rotate(t, 
			expand=False, fillcolor=0)) for t in rotation_ls])

		inputs = torch.cat((
			torch.as_tensor(peg_pred, dtype=torch.float32).unsqueeze(0),
			torch.as_tensor(hole_rots, dtype=torch.float32)), 0)

		outs = pose_model(inputs.unsqueeze(1).to(device))
		anchor_flatten = outs[0].view(1,-1)
		match_flatten = outs[1:].view(11,-1)
		dis = F.pairwise_distance(anchor_flatten.tile((11,1)), match_flatten)
		theta = rotation_ls[dis.argmin()]
		print('pred theta', theta)
	return theta


def get_dxy(o, device):
	with torch.no_grad():
		seg_pred = torch.as_tensor(o['gt'], dtype=torch.float32)
		outs = position_model(seg_pred.unsqueeze(1).to(device))
		# preds = outs.max(1)[1]
		preds = (outs[:,1,:,:].max()==outs[:,1,:,:]).type(torch.long)
		n,h,w = preds.size()
		preds_flatten = preds.view(n,-1)
		idx = np.argmax(preds_flatten.detach().cpu().numpy(), axis=1)
		dy, dx = idx // w, idx % w
		# 图像坐标系转世界坐标系
		dx = -(dx-10)/1000
		dy = (dy-10)/1000

		# get ground truth
		position_gts = get_position_gt(o)
		_, acc_position_5, _, _ = accuracy(
			preds.unsqueeze(1).detach().cpu().numpy(),
			np.expand_dims(position_gts, axis=1), thr=5)
		print('acc_position_5', acc_position_5, 'dxy', dx, dy, 'gt', o['dxy'])

		# plt.subplot(2,3,1)
		# plt.imshow(o['gt'][0])
		# plt.subplot(2,3,2)
		# plt.imshow(position_gts[0])
		# plt.subplot(2,3,3)
		# plt.imshow(preds[0].detach().cpu().numpy())
		# plt.subplot(2,3,4)
		# plt.imshow(outs[0][0].detach().cpu().numpy())
		# plt.subplot(2,3,5)
		# plt.imshow(outs[0][1].detach().cpu().numpy())
		# plt.show()

	return dx, dy, acc_position_5


def main():
	success_acc = 0
	success_acc_first = 0
	for i in range(100):
		o = venv.reset()
		dx = np.random.uniform(-0.005, 0.005)
		dy = np.random.uniform(-0.005, 0.005)
		# !!!!!!!!!!1
		dyaw = np.random.uniform(-10,10)
		# dyaw = 0
		o, r, d, info = venv.step([[dx, dy, dyaw]])

		# p.addUserDebugText("episode "+ str(i), [-1.1,0.,0], textSize=1, lifeTime=2)
		# time.sleep(2)
		print(i)

		for t in range(10):
			# seg_pred = seg_model(torch.as_tensor(o['img'], 
			# 	dtype=torch.float32).unsqueeze(0).to(args.device))
			# seg_pred = seg_pred.max(1)[1].cpu().detach().numpy()

			dx, dy, acc_position_5 = get_dxy(o, args.device)
			if t == 0:
				success_acc_first += acc_position_5

			# !!!!!!!!!!!1
			dyaw = get_dyaw(o, args.device)
			# dyaw = 0
			# print('i', i, 't', t, 'gt', o['dxy'], 'pred', dx, dy)
			# !!!!!!!!!!!!!!
			o, r, d, info = venv.step([[-dx/2, -dy/2, -dyaw]])
			# time.sleep(1)

			# !!!!!!!!!!!!!!1
			# if abs(o['dyaw']) <= 2:
			if abs(o['dxy'][0][0]) <= 0.001 and abs(o['dxy'][0][1]) <= 0.001 and abs(o['dyaw'][0])<=2:
			# if abs(o['dxy'][0][0]) <= 0.001 and abs(o['dxy'][0][1]) <= 0.001:
				# print(i, o['dyaw'], "success")
				print(i, o['dxy'], 'success', success_acc)
				dx, dy, acc_position_5 = get_dxy(o, args.device)
				success_acc += 1
				break

	print('success_acc', success_acc)
	print('success_acc_first', success_acc_first)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--seg_model_path', type=str, default='')
	parser.add_argument('--position_model_path', type=str, default='')
	parser.add_argument('--pose_model_path', type=str, default='')
	parser.add_argument('--device', type=str, default='cuda')
	args = parser.parse_args()

	envs = []
	# env = gym.make('gymEnv:peg-in-hole-v11', peg_type='square-concave1', seed=4, test_mode=True)
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-concave1', seed=2, test_mode=True))
	venv = SubprocVecEnv(envs)
	# seg_model = torch.load(args.seg_model_path, map_location=args.device)
	pose_model = torch.load(args.pose_model_path, map_location=args.device)
	position_model = torch.load(args.position_model_path, map_location=args.device)
	
	main()





# 先调整角度再调整位置
