import numpy as np
import torch
import argparse

def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()
env_group = parser.add_argument_group('environment')
env_group.add_argument('--env', type=str, default='close_loop_block_picking', help='The training environment')
env_group.add_argument('--robot', type=str, default='kuka', choices=['kuka', 'ur5', 'ur5_robotiq'], help='The robot to use in the environment')
env_group.add_argument('--num_objects', type=int, default=1, help='The number of objects in the environment.')
env_group.add_argument('--max_episode_steps', type=int, default=50, help='The maximal number of steps per episode')
env_group.add_argument('--action_sequence', type=str, default='pxyzr', choices=['pxyzr', 'pxyz'], help='The action space')
env_group.add_argument('--random_orientation', type=strToBool, default=True, help='Allow the environment to initialize with random orientation')
env_group.add_argument('--num_processes', type=int, default=5, help='The number of parallel environments to run')
env_group.add_argument('--render', type=strToBool, default=False, help='Render the PyBullet GUI or not')
env_group.add_argument('--workspace_size', type=float, default=0.3, help='Size of the workspace in meters')
env_group.add_argument('--heightmap_size', type=int, default=128, help='Size of the heightmap in pixels')
env_group.add_argument('--view_type', type=str, default='camera_center_xyz')
env_group.add_argument('--obs_type', type=str, default='pixel')

training_group = parser.add_argument_group('training')
training_group.add_argument('--algorithm', default='equi_sacfd', choices=['sac', 'sacfd', 'equi_sac', 'equi_sacfd',
                                                                          'ferm_sac', 'ferm_sacfd', 'rad_sac', 'rad_sacfd',
                                                                          'drq_sac', 'drq_sacfd'], help='The learning algorithm')
training_group.add_argument('--lr', type=float, default=1e-3, help='The learning rate')
training_group.add_argument('--actor_lr', type=float, default=None, help='The learning rate for the actor. If None, use lr')
training_group.add_argument('--critic_lr', type=float, default=None, help='The learning rate for the critic. If None, use lr')
training_group.add_argument('--gamma', type=float, default=0.99, help='The discount factor')
training_group.add_argument('--explore', type=int, default=0, help='The number of exploration steps. Not useful for sac algorithms')
training_group.add_argument('--fixed_eps', action='store_true', help='If true, fix the epsilon at the final value. Not useful for sac algorithms')
training_group.add_argument('--init_eps', type=float, default=1.0, help='The initial value of the epsilon schedule. Not useful for sac algorithms')
training_group.add_argument('--final_eps', type=float, default=0., help='The final value of the epsilon schedule. Not useful for sac algorithms')
training_group.add_argument('--training_iters', type=int, default=1, help='The number of training iterations per step')
training_group.add_argument('--training_offset', type=int, default=100, help='The minimal number of transitions to start training')
training_group.add_argument('--max_train_step', type=int, default=10000, help='The maximal number of training steps')
training_group.add_argument('--device_name', type=str, default='cuda', help='The device for PyTorch')
training_group.add_argument('--target_update_freq', type=int, default=100, help='The frequency of updating the target network. Not useful for sac algorithms')
training_group.add_argument('--save_freq', type=int, default=500, help='The frequency of logging')
training_group.add_argument('--load_model_pre', type=str, default=None, help='The directory to load model')
training_group.add_argument('--planner_episode', type=int, default=100, help='The number of demonstration episodes to gather before training')
training_group.add_argument('--note', type=str, default=None)
training_group.add_argument('--seed', type=int, default=None)
training_group.add_argument('--pre_train_step', type=int, default=0, help='The number of pre-training steps')
training_group.add_argument('--tau', type=float, default=1e-2, help='The soft update magnitude')
training_group.add_argument('--init_temp', type=float, default=1e-2, help='The initial temperature')
training_group.add_argument('--dpos', type=float, default=0.05, help='The maximal positional delta')
training_group.add_argument('--drot_n', type=int, default=4, help='The maximal rotational delta (pi/drot_n)')
training_group.add_argument('--demon_w', type=float, default=1, help='The weight of the demonstration loss')
training_group.add_argument('--equi_n', type=int, default=8, help='The order of the equivariant group for equivariant networks')
training_group.add_argument('--n_hidden', type=int, default=64, help='The hidden dimension of the network (only applicable for some networks)')
training_group.add_argument('--crop_size', type=int, default=128, help='The size of the image observation after cropping')
training_group.add_argument('--aug', type=strToBool, default=False, help='If true, perform RAD data augmentation at each sample step')
training_group.add_argument('--buffer_aug_type', type=str, choices=['se2', 'so2', 't', 'dqn_c4', 'so2_vec', 'shift', 'crop'], default='so2')
training_group.add_argument('--aug_type', type=str, choices=['se2', 'so2', 't', 'dqn_c4', 'so2_vec', 'shift', 'crop'], default='so2')
training_group.add_argument('--buffer_aug_n', type=int, default=4, help='The number of augmentation to perform in augmentation buffer (aug, per_expert_aug)')
training_group.add_argument('--expert_aug_n', type=int, default=0, help='The number of augmentation to perform for the expert data')

eval_group = parser.add_argument_group('eval')
eval_group.add_argument('--num_eval_processes', type=int, default=5, help='The number of parallel environments for evaluation')
eval_group.add_argument('--eval_freq', default=500, type=int, help='The evaluation frequency')
eval_group.add_argument('--num_eval_episodes', default=100, type=int, help='The number of evaluation episodes')

buffer_group = parser.add_argument_group('buffer')
buffer_group.add_argument('--buffer', default='normal', choices=['normal', 'per', 'expert', 'per_expert', 'aug', 'per_expert_aug'])
buffer_group.add_argument('--per_eps', type=float, default=1e-6, help='Epsilon parameter for PER')
buffer_group.add_argument('--per_alpha', type=float, default=0.6, help='Alpha parameter for PER')
buffer_group.add_argument('--per_beta', type=float, default=0.4, help='Initial beta parameter for PER')
buffer_group.add_argument('--per_expert_eps', type=float, default=1, help='The additional eps for expert data')
buffer_group.add_argument('--batch_size', type=int, default=64)
buffer_group.add_argument('--buffer_size', type=int, default=100000)

logging_group = parser.add_argument_group('logging')
logging_group.add_argument('--log_pre', type=str, default='output')
logging_group.add_argument('--log_sub', type=str, default=None)
logging_group.add_argument('--no_bar', action='store_true')
logging_group.add_argument('--time_limit', type=float, default=10000)
logging_group.add_argument('--load_sub', type=str, default=None)

args = parser.parse_args()
# env
random_orientation = args.random_orientation
env = args.env
num_objects = args.num_objects
max_episode_steps = args.max_episode_steps
action_sequence = args.action_sequence
num_processes = args.num_processes
num_eval_processes = args.num_eval_processes
render = args.render
robot = args.robot


workspace_size = args.workspace_size
workspace = np.asarray([[0.45-workspace_size/2, 0.45+workspace_size/2],
                        [0-workspace_size/2, 0+workspace_size/2],
                        [0.01, 0.25]])
heightmap_size = args.heightmap_size

heightmap_resolution = workspace_size/heightmap_size
action_space = [0, heightmap_size]
view_type = args.view_type
obs_type = args.obs_type

if env == 'close_loop_clutter_picking':
  tray = True
else:
  tray = False

######################################################################################
# training
algorithm = args.algorithm
lr = args.lr
actor_lr = args.actor_lr
critic_lr = args.critic_lr
if actor_lr is None:
    actor_lr = lr
if critic_lr is None:
    critic_lr = lr

gamma = args.gamma
explore = args.explore
fixed_eps = args.fixed_eps
init_eps = args.init_eps
final_eps = args.final_eps
training_iters = args.training_iters
training_offset = args.training_offset
max_train_step = args.max_train_step
device = torch.device(args.device_name)
target_update_freq = args.target_update_freq
save_freq = args.save_freq
planner_episode = args.planner_episode

load_model_pre = args.load_model_pre
note = args.note
seed = args.seed

tau = args.tau
init_temp = args.init_temp

demon_w = args.demon_w
equi_n = args.equi_n
n_hidden = args.n_hidden

crop_size = args.crop_size

aug = args.aug
aug_type = args.aug_type
buffer_aug_type = args.buffer_aug_type
buffer_aug_n = args.buffer_aug_n
expert_aug_n = args.expert_aug_n

# eval
eval_freq = args.eval_freq
num_eval_episodes = args.num_eval_episodes

# pre train
pre_train_step = args.pre_train_step

# buffer
buffer_type = args.buffer
per_eps = args.per_eps
per_alpha = args.per_alpha
per_beta = args.per_beta
per_expert_eps = args.per_expert_eps
batch_size = args.batch_size
buffer_size = args.buffer_size

# logging
log_pre = args.log_pre
log_sub = args.log_sub
no_bar = args.no_bar
time_limit = args.time_limit
load_sub = args.load_sub
if load_sub == 'None':
    load_sub = None

dpos = args.dpos
drot = np.pi/args.drot_n

if algorithm == 'sac':
  alg = 'sac'
  model = 'cnn'
elif algorithm == 'sacfd':
  alg = 'sacfd'
  model = 'cnn'

elif algorithm == 'equi_sac':
  alg = 'sac'
  model = 'equi_both'
elif algorithm == 'equi_sacfd':
  alg = 'sacfd'
  model = 'equi_both'

elif algorithm == 'ferm_sac':
  alg = 'curl_sac'
  model = 'cnn'
  heightmap_size = 142
  heightmap_resolution = workspace_size / heightmap_size
elif algorithm == 'ferm_sacfd':
  alg = 'curl_sacfd'
  model = 'cnn'
  heightmap_size = 142
  heightmap_resolution = workspace_size / heightmap_size

elif algorithm == 'rad_sac':
  alg = 'sac'
  model = 'cnn'
  heightmap_size = 142
  heightmap_resolution = workspace_size / heightmap_size
  aug = True
  aug_type = 'crop'
elif algorithm == 'rad_sacfd':
  alg = 'sacfd'
  model = 'cnn'
  heightmap_size = 142
  heightmap_resolution = workspace_size / heightmap_size
  aug = True
  aug_type = 'crop'

elif algorithm == 'drq_sac':
  alg = 'sac_drq'
  model = 'cnn'
  aug_type = 'shift'
elif algorithm == 'drq_sacfd':
  alg = 'sacfd_drq'
  model = 'cnn'
  aug_type = 'shift'

######################################################################################
env_config = {'workspace': workspace, 'max_steps': max_episode_steps, 'obs_size': heightmap_size,
              'action_sequence': action_sequence, 'render': render, 'num_objects': num_objects,
              'random_orientation':random_orientation, 'robot': robot, 'view_type': view_type,
              'obs_type': obs_type, 'close_loop_tray': tray}
planner_config = {'random_orientation':random_orientation, 'dpos': dpos, 'drot': drot}
if seed is not None:
    env_config['seed'] = seed
######################################################################################
hyper_parameters = {}
for key in sorted(vars(args)):
    hyper_parameters[key] = vars(args)[key]

for key in hyper_parameters:
    print('{}: {}'.format(key, hyper_parameters[key]))