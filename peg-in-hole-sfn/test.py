import numpy as np 
import gym
import gymEnv
from utils.vec_env.subproc_vec_env import SubprocVecEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
np.set_printoptions(threshold=np.inf)

import skimage.data

from scipy.spatial.transform import Rotation as R
from skimage import data
from PIL import Image
import cv2
import pybullet as p
import time
import os

# from dataloader import UR5Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision

# img = data.chelsea().astype(int)
# print(img.shape)
# print(type(img))
# plt.imshow(img)
# plt.show()
# exit(0)
# 


rotation_ls_10 = np.arange(-30,40,10)
rotation_ls_6 = np.arange(-15,16,6)
rotation_ls_2 = np.arange(-7,8,2)
rotation_ls = np.array(list(set(np.concatenate((
    rotation_ls_10, 
    rotation_ls_6, 
    rotation_ls_2),0))))

# rotation_ls_10 = np.arange(-30,31,3)
# rotation_ls_6 = np.arange(-15,16,6)
# rotation_ls_2 = np.arange(-6,7,3)
# rotation_ls = np.array(list(set(np.concatenate((
#     rotation_ls_10, 
#     rotation_ls_6, 
#     rotation_ls_2),0))))

print(np.sort(rotation_ls))
print(rotation_ls_2)
print(rotation_ls_6)
print(rotation_ls_10)
exit(0)



def rot_img(x, theta):
	theta = torch.tensor(theta)
	rot_mat = torch.tensor([
		[torch.cos(theta), -torch.sin(theta), 0],
		[torch.sin(theta), torch.cos(theta), 0]])
	rot_mat = rot_mat[None, ...].repeat(x.shape[0],1,1)
	grid = F.affine_grid(rot_mat, x.size(), align_corners=True)
	x = F.grid_sample(x, grid, align_corners=True)
	return x


#im should be a 4D tensor of shape B x C x H x W with type dtype, range [0,255]:
img1 = data.chelsea().astype(int)
img2 = data.chelsea().astype(int)
im1 = torch.tensor(img1, dtype=torch.float).permute(2,0,1).unsqueeze(0)
im2 = torch.tensor(img2, dtype=torch.float).permute(2,0,1).unsqueeze(0)
im = torch.cat((im1,im2),0)

#Rotation by np.pi/2 with autograd support:
rotated_im = rot_img(im, np.pi/4) # Rotate image by 90 degrees.

plt.subplot(2,2,1)
plt.imshow(im[0].permute(1,2,0)/255) #To plot it im should be 1 x C x H x W
plt.subplot(2,2,2)
plt.imshow(im[1].permute(1,2,0)/255)
plt.subplot(2,2,3)
plt.imshow(rotated_im[0].permute(1,2,0)/255)
plt.subplot(2,2,4)
plt.imshow(rotated_im[1].permute(1,2,0)/255)
plt.show()
exit(0)



class Network(nn.Module):
	"""docstring for Network"""
	def __init__(self):
		super(Network, self).__init__()
		self.conv = nn.Conv2d(3,3,3,1,1)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv(x))
		return x


model = Network()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
img = data.chelsea().astype(int)
im = torch.tensor(img, dtype=torch.float).permute(2,0,1).unsqueeze(0)
l = torch.ones(2,3,300,451)

for i in range(10):
	b = model(im)
	c = rot_img(b, np.pi/4, torch.FloatTensor)
	d = torch.cat((b,c),0)
	loss = criterion(d.view(2,-1), l.view(2,-1))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print(loss)

exit(0)

# transform: 250~700, 400~900

trainSet = UR5Dataset(root='/home/xieliang/data/ur5', 
	annotation_path='/home/xieliang/data/ur5/annotation_train.csv')
testSet = UR5Dataset(root='/home/xieliang/data/ur5', 
	annotation_path='/home/xieliang/data/ur5/annotation_test.csv')
trainLoader = DataLoader(
	trainSet, batch_size=1, shuffle=True, 
	num_workers=4, pin_memory=True)
testLoader = DataLoader(
	testSet, batch_size=10, shuffle=False,
	num_workers=4, pin_memory=True)

for batch_idx, batch in enumerate(trainLoader):
	print(batch_idx)
	rgb = cv2.cvtColor(batch['img'][0].numpy(),cv2.COLOR_BGR2RGB)
	plt.subplot(1,2,1)
	plt.imshow(batch['seg_gt'][0].numpy())
	plt.subplot(1,2,2)
	plt.imshow(rgb)
	plt.show()
	exit(0)

print(len(trainSet))
print(len(testSet))
exit(0)






#  1）cam2base求逆， 2）单位mm -> m
base = np.array([0.3688, -0.2317, -0.1493, 1])
# base = np.array([0.5779, -0.2426, -0.0483, 1])
# base = np.array([0.4488, -0.2913, -0.1423, 1])
extrinsic = np.array([[-0.53092894, -0.69699066, -0.48199428,  0.73024807],
       [-0.51010983,  0.71705235, -0.47499883,  0.03853732],
       [ 0.67668488, -0.00632061, -0.73624563,  0.21127942],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
print('inv', np.linalg.inv(extrinsic))
extrinsic = np.linalg.inv(extrinsic)
camera = np.matmul(extrinsic, base)[:-1].reshape(-1,1)
print(camera)
intrinsic = np.array([[913.12513623,   0.        , 648.83544935],
					[  0.        , 913.69982734, 375.72440754],
					[  0.        ,   0.        ,   1.        ]])
pixel = np.matmul(intrinsic,camera)/camera[-1]
print(pixel)

img = plt.imread('/home/xieliang/桌面/ttttt/3/image.jpg')
plt.imshow(img)
plt.show()
exit(0)



torch.manual_seed(0)
a = torch.randn((6))
b = torch.tensor([-1])
print(a)
c = torch.cat((a,b),0).numpy()
c.sort()
print(c[::-1])
idx = np.argwhere(c[::-1]==b.numpy())
print(idx)
exit(0)

img_ori = data.horse()
img1 = Image.fromarray(np.uint8(img_ori))
plt.subplot(3,4,1)
plt.imshow(img1, cmap=plt.cm.gray)
img2 = img1.rotate(30, expand=False, fillcolor=1)
plt.subplot(3,4,2)
plt.imshow(img2, cmap=plt.cm.gray)
img3 = img1.rotate(60, expand=False, fillcolor=1)
plt.subplot(3,4,3)
plt.imshow(img3, cmap=plt.cm.gray)

plt.show()
exit(0)

random.seed(0)
b = list(range(9))
print(b)
random.shuffle(b)
print(b)
for idx, i in enumerate(range(0, len(b), 3)):
	print(idx, b[i:i+3])
exit(0)

# n_class = 3
# true = np.array([[0,1,0],[0,1,1],[0,0,0]]).flatten()
# pred = np.array([[0,0,0],[2,1,1],[0,0,0]]).flatten()
# hist = np.bincount(n_class*true+pred, minlength=n_class ** 2).reshape(n_class, n_class)
# print('hist', hist)
# acc = np.diag(hist).sum() / hist.sum()
# acc_cls = np.nanmean(np.diag(hist) / hist.sum(axis=1))
# iu = np.diag(hist) / (hist.sum(axis=0)+hist.sum(axis=1)-np.diag(hist))
# meanIU = np.nanmean(iu)
# freq = hist.sum(axis=1) / hist.sum()
# fwIU = (freq[freq > 0] * iu[freq > 0]).sum() 
# print('acc', acc)
# print('acc_cls', acc_cls)
# print('meanIU', meanIU)
# print('fwIU', fwIU)
# exit(0)

# img = np.zeros((224,224))
# cv2.ellipse(img, (112,112), (60,60), 0, 30, 330, (1,0,0), -1)