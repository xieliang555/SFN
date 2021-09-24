  
if __name__ == '__main__':
	import gym
	import gymEnv
	print(gymEnv.__path__)
	import os
	import signal, psutil
	import time
  
	import algos.pytorch.fcn.visualize_pose_8 as pose
	from utils.logx import EpochLogger
	from utils.vec_env.subproc_vec_env import SubprocVecEnv
	from utils.run_utils import setup_logger_kwargs 
  
	import os
	# os.environ['CUDA_VISIBLE_DEVICES']= "2"
 
	import numpy as np
	np.set_printoptions(threshold=np.inf)
 
	import argparse
	parser = argparse.ArgumentParser()
	# batch_size has to be the divisor of the buffer_size
	parser.add_argument('--seed', '-s', type=int, default=1)
	parser.add_argument('--steps', type=int, default=200)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--model_path', type=str, default='')
	args = parser.parse_args()
 
	envs = []
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v8', peg_type='square-convex4', seed=args.seed))
	venv = SubprocVecEnv(envs)
	pose.test_pose(venv=venv,nenv=len(envs), seed=args.seed, 
		local_steps_per_epoch=args.steps//len(envs),
		device=args.device, model_path=args.model_path)
 


