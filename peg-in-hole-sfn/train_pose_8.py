 
if __name__ == '__main__':
	import gym
	import gymEnv
	print(gymEnv.__path__)
	import os
	import signal, psutil
	import time
  
	import algos.pytorch.fcn.pose_8 as pose
	from utils.logx import EpochLogger
	from utils.vec_env.subproc_vec_env import SubprocVecEnv
	from utils.run_utils import setup_logger_kwargs 
  
	import os
	# os.environ['PYOPENGL_PLATFORM'] = 'egl'
	# os.environ['CUDA_VISIBLE_DEVICES']= "1"
 
	import numpy as np
	np.set_printoptions(threshold=np.inf)
 
	import argparse
	parser = argparse.ArgumentParser()
	# batch_size has to be the divisor of the buffer_size
	parser.add_argument('--seed', '-s', type=int, default=1)
	parser.add_argument('--steps', type=int, default=200)
	parser.add_argument('--epochs', type=int, default=15)
	parser.add_argument('--iterates', type=int, default=1000)
	parser.add_argument('--exp_name', type=str, default='pose_8_resumes_test')
	parser.add_argument('--data_dir', type=str, default='/home/xieliang/models/peg-in-hole-v1')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--resume', action='store_true', default=False)
	parser.add_argument('--model_path', type=str, default='')
	args = parser.parse_args()
 
	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, args.data_dir)
	envs = []
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-triangle', seed=args.seed))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-square', seed=args.seed+1))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-pentagon', seed=args.seed+2))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-hexagon', seed=args.seed+3))
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-fillet1', seed=args.seed+4))
	venv = SubprocVecEnv(envs)
	pose.train_pose(venv=venv,nenv=len(envs), seed=args.seed, 
		local_steps_per_epoch=args.steps//len(envs), epochs=args.epochs, logger_kwargs=logger_kwargs, 
		device=args.device, resume=args.resume, model_path=args.model_path, iterates=args.iterates)
 


