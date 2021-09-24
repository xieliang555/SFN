import numpy as np 
import gym
import gymEnv
from utils.vec_env.subproc_vec_env import SubprocVecEnv
import torch
import torch.nn as nn
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



img1 = np.zeros((200,200), dtype=np.uint8)
pts1 = np.array([[10,10],[100,10],[100,100],[10,89]], dtype=np.int32)
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img, pts, isClosed=True, color=(1,0,0), thickness=1)
cv2.fillPoly(img1, [pts1], color=(1, 0, 0))
img2 = np.zeros((200,200), dtype=np.uint8)
pts2 = np.array([[20,20],[120,20],[120,120],[20,189]], dtype=np.int32)
cv2.fillPoly(img2, [pts2], color=(2,0,0))
img3 = cv2.bitwise_xor(img1, img2)
plt.subplot(1,3,1)
plt.imshow(img1)
plt.subplot(1,3,2)
plt.imshow(img2)
plt.subplot(1,3,3)
plt.imshow(img3)
plt.show()
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


rotation_ls_10 = np.arange(-30,31,3)
rotation_ls_6 = np.arange(-15,16,6)
rotation_ls_2 = np.arange(-6,7,3)
rotation_ls = np.array(list(set(np.concatenate((
    rotation_ls_10, 
    rotation_ls_6, 
    rotation_ls_2),0))))
print(np.sort(rotation_ls))
print(rotation_ls_2)
print(rotation_ls_6)
print(rotation_ls_10)
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