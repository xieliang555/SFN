  
if __name__ == '__main__':
	import gym
	import gymEnv
	print(gymEnv.__path__)
	import os
	import signal, psutil
	import time
  
	import algos.pytorch.fcn.position_11 as position
	from utils.logx import EpochLogger
	from utils.vec_env.subproc_vec_env import SubprocVecEnv
	from utils.run_utils import setup_logger_kwargs 
  
	import os
	os.environ['PYOPENGL_PLATFORM'] = 'egl'
	# os.environ['CUDA_VISIBLE_DEVICES']= "1"
  
	import numpy as np
	np.set_printoptions(threshold=np.inf)
 
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', '-s', type=int, default=1)
	parser.add_argument('--steps', type=int, default=200)
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--iterates', type=int, default=1000)
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--exp_name', type=str, default='position_11_ff_resume_test')
	parser.add_argument('--data_dir', type=str, default='/home/xieliang/models/peg-in-hole-v1')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--resume', action='store_true', default=False)
	parser.add_argument('--position_model_path', type=str, default='')
	parser.add_argument('--seg_model_path', type=str, default='')
	args = parser.parse_args()
 
	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, args.data_dir)
	envs = []
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-triangle', seed=args.seed))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-triangle', seed=args.seed+1))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-pentagon', seed=args.seed+2))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-hexagon', seed=args.seed+3))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-diamond', seed=args.seed+4))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-trapezoid', seed=args.seed+5))
	envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-concave1', seed=args.seed+6))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-concave2', seed=args.seed+7))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-convex1', seed=args.seed+8))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-convex2', seed=args.seed+9))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-convex3', seed=args.seed+10))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-convex4', seed=args.seed+11))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-fillet2', seed=args.seed+12))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-fillet3', seed=args.seed+13))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-fillet4', seed=args.seed+14))
	# envs.append(lambda:gym.make('gymEnv:peg-in-hole-v11', peg_type='square-fillet1', seed=args.seed+15))
	venv = SubprocVecEnv(envs)
	position.train_position(venv=venv, nenv=len(envs), seed=args.seed, 
		local_steps_per_epoch=args.steps//len(envs), epochs=args.epochs, 
		logger_kwargs=logger_kwargs, device=args.device, resume=args.resume, 
		position_model_path=args.position_model_path, seg_model_path=args.seg_model_path, 
		iterates=args.iterates, batch_size=args.batch_size)
 




# 仿真中随机改变相机位姿，验证实物测试效果不好是否是相机位姿的原因
