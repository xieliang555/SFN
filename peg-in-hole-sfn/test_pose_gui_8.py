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

parser = argparse.ArgumentParser()
parser.add_argument('--seg_model_path', type=str, default='')
parser.add_argument('--pose_model_path', type=str, default='')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

env = gym.make('gymEnv:peg-in-hole-v8', peg_type='square-diamond', seed=4, gui_mode=True)
# seg_model = torch.load(args.seg_model_path, map_location=args.device)
pose_model = torch.load(args.pose_model_path, map_location=args.device)

success_acc = 0
for i in range(10):
	o = env.reset()
	theta = np.random.uniform(-15,15)
	o, r, d, info = env.step(theta)

	p.addUserDebugText("episode "+ str(i), [-1.1,0.,0], textSize=1, lifeTime=2)
	# time.sleep(2)

	success = 0
	for t in range(10):
		with torch.no_grad():
			# seg_pred = seg_model(torch.as_tensor(o['img'], dtype=torch.float32).unsqueeze(0).to(args.device))
			# seg_pred = seg_pred.max(1)[1].cpu().detach().numpy()
			seg_pred = o['gt']
			peg_pred = np.float32(seg_pred==1)
			hole_pred = np.float32(seg_pred==2)

			rotation_ls = np.array([-10,-8,-6,-4,-2, 0, 2,4,6,8,10])
			idx_pos = abs(o['dtheta']-rotation_ls).argmin()
			theta_pos = rotation_ls[idx_pos]
			print('dtheta/gt dtheta', o['dtheta'],'/',theta_pos)

			# idx_neg = [i for i in range(len(rotation_ls)) if i != idx_pos]
			# theta_neg = rotation_ls[idx_neg]
			# base_mask_pos = np.array([np.array(Image.fromarray(np.uint8(hole_pred[0])).rotate(
			# 	theta_pos, expand=False, fillcolor=0))])

			hole_rots = np.array([np.array(Image.fromarray(np.uint8(hole_pred)).rotate(t, 
				expand=False, fillcolor=0)) for t in rotation_ls])


			inputs = torch.cat((
				torch.as_tensor(peg_pred, dtype=torch.float32).unsqueeze(0),
				torch.as_tensor(hole_rots, dtype=torch.float32)), 0)

			outs = pose_model(inputs.unsqueeze(1).to(args.device))
			anchor_flatten = outs[0].view(1,-1)
			match_flatten = outs[1:].view(11,-1)
			dis = F.pairwise_distance(anchor_flatten.tile((11,1)), match_flatten)
			theta = rotation_ls[dis.argmin()]
			print('pred theta', theta)

			o, r, d, info = env.step(-theta)
			# time.sleep(1)

			if abs(o['dtheta']) <= 2:
				print(i, o['dtheta'], "success")
				success = 1
				success_acc += 1
				p.addUserDebugText("episode "+ str(i) + " Success", [-1.1,0.,0], textSize=1, lifeTime=2)
				# time.sleep(2)
				break


	if not success:
		p.addUserDebugText("episode "+ str(i) + " Fail", [-1.1,0.,0], textSize=1, lifeTime=2)
		# time.sleep(2)

print('success acc', success_acc)

			# plt.subplot(3,4,1)
			# plt.imshow(o['img'].transpose((1,2,0)))
			# plt.subplot(3,4,2)
			# plt.imshow(seg_pred[0])
			# plt.subplot(3,4,3)
			# plt.imshow(peg_pred[0])
			# plt.subplot(3,4,4)
			# plt.imshow(hole_pred[0])

			# plt.subplot(3,4,5)
			# plt.imshow(hole_rots[0])
			# plt.subplot(3,4,6)
			# plt.imshow(hole_rots[1])
			# plt.subplot(3,4,7)
			# plt.imshow(hole_rots[2])
			# plt.subplot(3,4,8)
			# plt.imshow(hole_rots[3])

			# plt.subplot(3,4,9)
			# plt.imshow(hole_rots[4])
			# plt.subplot(3,4,10)
			# plt.imshow(hole_rots[5])
			# plt.subplot(3,4,11)
			# plt.imshow(hole_rots[6])

			# plt.show()