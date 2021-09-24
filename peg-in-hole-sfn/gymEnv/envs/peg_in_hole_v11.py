import gym
import pybullet as p
import os
import math
import numpy as np
import pkgutil
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
from utils.pid_control import PID
from scipy.spatial.transform import Rotation as R
   
import numpy as np
import random
import pybullet_data
import pyrender
import trimesh

 
   
class PegInHole(gym.Env):
	"""docstring for PegInHole"""
	def __init__(self, peg_type=None, seed=0, test_mode=False):
		super(PegInHole, self).__init__()
		# connect to pybullet
		if 0:
			self.id = p.connect(p.GUI)
		else:
			self.id = p.connect(p.DIRECT)
			# egl = pkgutil.get_loader('eglRenderer')
			# if egl:
			# 	p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
			# else:
			# 	p.loadPlugin("eglRendererPlugin")

		# load URDF
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.panda_peg_id = p.loadURDF(os.path.join(os.path.dirname(
			os.path.realpath(__file__)),"complex/"+peg_type+"/peg/peg.urdf"), 
			baseOrientation=p.getQuaternionFromEuler([0,0,math.pi]),
			basePosition=[-0.5,0,0], globalScaling=1, useFixedBase=1)
		self.hole_id = p.loadURDF(os.path.join(os.path.dirname(
			os.path.realpath(__file__)), "complex/"+peg_type+"/base/base.urdf"), 
			baseOrientation=p.getQuaternionFromEuler([0,0,0]),
			basePosition=[-1,0,0], globalScaling=1, useFixedBase=1)


		# define space
		self.img_width, self.img_height = 1280, 720
		self.observation_space = gym.spaces.Box(
			low=np.float32(-1.0), high=np.float32(1.0), 
			shape=(1,200,250), dtype=np.float32)
		self.action_space = gym.spaces.Discrete(4)

		# force-torque sensor & camera setup
		self.ftJointIndex = 11
		p.enableJointForceTorqueSensor(self.panda_peg_id, self.ftJointIndex, 1)
		p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-135, 
			cameraPitch=-30, cameraTargetPosition=[-1,0,0])

		# other setup
		self.pandaNumDofs = 7
		self.endEffectorIndex = 12
		np.random.seed(seed)
		random.seed(seed)
		self.peg_type = peg_type
		self.test_mode = test_mode

		# set camera extrinsic and intrinsic matrix
		self.extrinsic = self.get_homogenous_matrix(
			R.from_euler('zx', [180,-45], degrees=True).as_matrix(), 
			[0,0.1,0.1])

		p.addUserDebugLine([-1,0.1,0.1],R.from_euler('zx',[180,-45], degrees=True).apply([0.1,0,0])+[-1,0.1,0.1], lineColorRGB=[1,0,0])
		p.addUserDebugLine([-1,0.1,0.1],R.from_euler('zx',[180,-45], degrees=True).apply([0,0.1,0])+[-1,0.1,0.1], lineColorRGB=[0,1,0])
		p.addUserDebugLine([-1,0.1,0.1],R.from_euler('zx',[180,-45], degrees=True).apply([0,0,0.1])+[-1,0.1,0.1], lineColorRGB=[0,0,1])

		# # compute extrinsic matrix using look-at method
		# c = np.array([0,0.1,0.1]) # camera eye position
		# pp = np.array([0,0,0]) # camera target position
		# u = np.array([0,-1,0]) # up vector
		# l = pp-c 
		# l_norm = l/np.linalg.norm(l)
		# s = np.cross(l_norm,u)
		# s_norm = s/np.linalg.norm(s)
		# u_ = np.cross(s_norm,l_norm)
		# extrinsic = np.stack((s_norm,u_,-l_norm))
		# self.extrinsic = self.get_homogenous_matrix(extrinsic, [0,0.1,0.1])
		# # # print('extrinsic', extrinsic)
		
		self.intrinsic = np.array(
			[self.img_height/(2*math.tan(math.pi/8)), 0, self.img_width/2, 0,
			0, self.img_height/(2*math.tan(math.pi/8)), self.img_height/2, 0,
			0, 0, 1, 0]).reshape(3,4)
		# for gui mode
		self.intrinsic_3plus3 = np.array(
			[self.img_height/(2*math.tan(math.pi/8)), 0, self.img_width/2,
			0, self.img_height/(2*math.tan(math.pi/8)), self.img_height/2,
			0, 0, 1]).reshape(3,3)
		

	def robot_init(self):
		self.jointPositions=[
			0.327, 0.369, -0.293, -2.383, 0.261, 2.726, 2.17, 0.02, 0.02]
		index = 0
		for j in range(p.getNumJoints(self.panda_peg_id)):
			p.changeDynamics(self.panda_peg_id, j, linearDamping=0, angularDamping=0)
			info = p.getJointInfo(self.panda_peg_id, j)
			jointName = info[1]
			jointType = info[2]
			if jointType == p.JOINT_PRISMATIC:
				p.resetJointState(self.panda_peg_id, j, self.jointPositions[index])
				index=index+1 
			if jointType == p.JOINT_REVOLUTE:
				p.resetJointState(self.panda_peg_id, j, self.jointPositions[index])
				index=index+1


	def position_control(self, num_step, target_pos, target_orn):
		for t in range(num_step):
			jointPoses = p.calculateInverseKinematics(
				self.panda_peg_id, self.endEffectorIndex, target_pos, 
				p.getQuaternionFromEuler(target_orn*math.pi/180))
			for i in range(self.pandaNumDofs):
				p.setJointMotorControl2(self.panda_peg_id, i, 
						p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
			for i in [9,10]:
				p.setJointMotorControl2(self.panda_peg_id,i, p.POSITION_CONTROL, 0.02, force=10)
			p.stepSimulation()

	def get_homogenous_matrix(self, rotation_matrix, translate_vector):
		home_matrix = np.eye(4)
		home_matrix[:3,:3] = rotation_matrix
		home_matrix[:3,3] = translate_vector
		return home_matrix

	def get_mask(self):
		scene = pyrender.Scene()
		# add mesh 
		tm = trimesh.load('./gymEnv/envs/complex/'+self.peg_type+'/mask.obj') 
		mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
		scene.add(mesh, pose=np.eye(4))
		# add camera
		camera = pyrender.IntrinsicsCamera(
			fx=self.img_height/(2*math.tan(math.pi/8)), 
			fy=self.img_height/(2*math.tan(math.pi/8)), 
			cx=self.img_width/2, cy=self.img_height/2)
		scene.add(camera, pose=self.extrinsic)
		# add light
		light = pyrender.SpotLight(color=np.ones(3), intensity=2, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
		scene.add(light, pose=self.extrinsic)
		r = pyrender.OffscreenRenderer(self.img_width, self.img_height)
		color, depth = r.render(scene)
		return color


	def reset(self):
		self.robot_init()
		self.mask = self.get_mask()
		self.init_orientation =  np.array([180, 0, 90])
		self.init_position = np.array([-1, 0, 0.002])
		self.position_control(100, self.init_position, self.init_orientation)

		x,y,z = -1-0.01,0.01,0
		p.addUserDebugLine([x-0.01,y,z],[x+0.01,y,z])
		p.addUserDebugLine([x,y-0.01,z],[x,y+0.01,z])
		p.addUserDebugLine([x,y,z-0.01],[x,y,z+0.01])

		self.peg_dx_acc = 0.0
		self.peg_dy_acc = 0.0
		self.peg_dyaw_acc = 0.0

		# # test pose transformation
		# eePos, eeOrn = p.getLinkState(self.panda_peg_id, self.endEffectorIndex)[0:2]
		# p.addUserDebugLine(eePos, R.from_quat(eeOrn).apply([0.1,0,0])+eePos, lineColorRGB=[1,0,0])
		# p.addUserDebugLine(eePos, R.from_quat(eeOrn).apply([0,0.1,0])+eePos, lineColorRGB=[0,1,0])
		# p.addUserDebugLine(eePos, R.from_quat(eeOrn).apply([0,0,0.1])+eePos, lineColorRGB=[0,0,1])
		# ori_euler = R.from_quat(eeOrn).as_euler('xyz', degrees=True)
		# print('ori_euler', ori_euler)
		# TCP_RT = self.get_homogenous_matrix(R.from_quat(eeOrn).as_matrix(), eePos)
		# target2tcp = self.get_homogenous_matrix(R.from_euler('z', 90, degrees=True).as_matrix(), [0,0,0.1])
		# target2base = TCP_RT.dot(target2tcp)
		# p.addUserDebugLine(target2base[:3,3], R.from_matrix(target2base[:3,:3]).apply([0.1,0,0])+target2base[:3,3], lineColorRGB=[1,0,0])
		# p.addUserDebugLine(target2base[:3,3], R.from_matrix(target2base[:3,:3]).apply([0,0.1,0])+target2base[:3,3], lineColorRGB=[0,1,0])
		# p.addUserDebugLine(target2base[:3,3], R.from_matrix(target2base[:3,:3]).apply([0,0,0.1])+target2base[:3,3], lineColorRGB=[0,0,1])




		img, gt= self.render()
		obs = {'img':img, 'gt':gt, 'dxy':[0,0], 'dyaw':0}
		return obs


	def render(self):
		cameraEyePos = p.getBasePositionAndOrientation(
			self.hole_id)[0]+np.array([0,0.1,0.1])
		cameraTargetPos = p.getBasePositionAndOrientation(self.hole_id)[0]
		upVector = [0,-1,0]

		viewMatrix = p.computeViewMatrix(cameraEyePos, cameraTargetPos, upVector)
		# 45指的fovy, 16/9用于计算fovx
		projectionMatrix = p.computeProjectionMatrixFOV(45, 16/9, 0.001, 10)
		_, _, rgb, _, _ = p.getCameraImage(
			self.img_width, self.img_height, 
			viewMatrix, projectionMatrix, shadow=1, 
			lightDirection=[np.random.uniform(-5,5), 
							np.random.uniform(-5,5), 
							np.random.uniform(1,10)],
			renderer=p.ER_TINY_RENDERER)
		
		rgb_img = rgb[:,:,0:3]
		hole_mask = cv2.inRange(self.mask,
			lowerb=np.array([0,0,0], dtype = "uint8"),
			upperb=np.array([254,254,254], dtype = "uint8"))
		peg_mask = cv2.inRange(rgb_img, 
			lowerb=np.array([0,0,0], dtype="uint8"),
			upperb=np.array([0,255,0], dtype="uint8"))
		intersection = cv2.bitwise_and(peg_mask, hole_mask)
		diff_mask = cv2.bitwise_xor(hole_mask, intersection)
		gt = cv2.bitwise_or(peg_mask/255, diff_mask/255*2)

		# # random rotate
		# random_rot = np.random.uniform(-5,5)
		# M = cv2.getRotationMatrix2D((640,360), random_rot, 1)
		# rgb_img = cv2.warpAffine(rgb_img, M, (1280, 720))
		# gt = cv2.warpAffine(gt, M, (1280, 720)).astype(np.long)

		# random crop 
		img = np.transpose(rgb_img, (2,0,1))
		if self.test_mode:
			# !!!!!!!!!11
			random_xy = [0,0]
			# random_xy = np.random.uniform(0,50,2)
		else:
			# !!!!!!!!!!11
			random_xy = [0,0]
			# random_xy = np.random.uniform(0,50,2)
		y_center = int(self.img_height // 2 + random_xy[1])
		x_center = int(self.img_width // 2 + random_xy[0])
		img = img[:, y_center-100:y_center+100, x_center-125:x_center+125]
		gt = gt[y_center-100:y_center+100, x_center-125:x_center+125]

		return img, gt


	def step(self, action=[0.0 ,0.0, 0.0]):
		# action: [dx, dy, dyaw]
		if not self.test_mode:
			# !!!!!!!! 0.01 -> 0.005
			dx = np.random.uniform(-0.01,0.01)
			dy = np.random.uniform(-0.01,0.01)
			dyaw = np.random.uniform(-10,10)
		else:
			self.peg_dx_acc += action[0]
			self.peg_dy_acc += action[1]
			self.peg_dyaw_acc += action[2]
			dx = self.peg_dx_acc
			dy = self.peg_dy_acc
			dyaw = self.peg_dyaw_acc

		target_pos = self.init_position + [dx, dy, 0]
		target_orn = self.init_orientation + [0, 0, dyaw]
		self.position_control(100, target_pos, target_orn)
		
		endPos = p.getLinkState(self.panda_peg_id, self.endEffectorIndex)[0]
		dxy = [endPos[0]+1, endPos[1]]
		img, gt = self.render()
		obs = {'img':img, 'gt':gt, 'dxy':dxy, 'dyaw':dyaw}
		return obs, 0, False, {}


	def close(self):
		p.disconnect()






