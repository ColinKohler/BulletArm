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
env_group.add_argument('--env', type=str, default='block_stacking', help='block_picking, block_stacking, brick_stacking, '
                                                                         'brick_inserting, block_cylinder_stacking')
env_group.add_argument('--robot', type=str, default='kuka')
env_group.add_argument('--num_objects', type=int, default=-1)
env_group.add_argument('--max_episode_steps', type=int, default=-1)
env_group.add_argument('--action_sequence', type=str, default='xyrp')
env_group.add_argument('--random_orientation', type=strToBool, default=True)
env_group.add_argument('--num_processes', type=int, default=5)
env_group.add_argument('--render', type=strToBool, default=False)
env_group.add_argument('--workspace_size', type=float, default=0.4)
env_group.add_argument('--heightmap_size', type=int, default=128)
env_group.add_argument('--patch_size', type=int, default=24)
env_group.add_argument('--in_hand_mode', type=str, default='raw', choices=['raw', 'proj'])

training_group = parser.add_argument_group('training')
training_group.add_argument('--alg', default='dqn')
training_group.add_argument('--model', type=str, default='resucat')
training_group.add_argument('--num_rotations', type=int, default=16)
training_group.add_argument('--half_rotation', type=strToBool, default=True)
training_group.add_argument('--lr', type=float, default=1e-4)
training_group.add_argument('--gamma', type=float, default=0.95)
training_group.add_argument('--explore', type=int, default=0)
training_group.add_argument('--fixed_eps', action='store_true')
training_group.add_argument('--init_eps', type=float, default=1.0)
training_group.add_argument('--final_eps', type=float, default=0.)
training_group.add_argument('--training_iters', type=int, default=1)
training_group.add_argument('--training_offset', type=int, default=100)
training_group.add_argument('--max_train_step', type=int, default=50000)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--target_update_freq', type=int, default=100)
training_group.add_argument('--save_freq', type=int, default=500)
training_group.add_argument('--load_model_pre', type=str, default=None)
training_group.add_argument('--sl', action='store_true')
training_group.add_argument('--planner_episode', type=int, default=0)
training_group.add_argument('--note', type=str, default=None)
training_group.add_argument('--seed', type=int, default=None)
training_group.add_argument('--perlin', type=float, default=0.0)
training_group.add_argument('--gaussian', type=float, default=0.0)
training_group.add_argument('--load_n', type=int, default=1000000)
training_group.add_argument('--expert_aug_n', type=int, default=9)
training_group.add_argument('--expert_aug_d4', action='store_true')
training_group.add_argument('--fill_buffer_deconstruct', action='store_true')
training_group.add_argument('--pre_train_step', type=int, default=0)
training_group.add_argument('--num_zs', type=int, default=36)
training_group.add_argument('--min_z', type=float, default=0.02)
training_group.add_argument('--max_z', type=float, default=0.20)
training_group.add_argument('--q2_model', type=str, default='cnn')
training_group.add_argument('--equi_n', type=int, default=4)
training_group.add_argument('--aug', type=strToBool, default=False)
training_group.add_argument('--aug_type', type=str, choices=['se2', 'cn', 't', 'shift'], default='se2')

eval_group = parser.add_argument_group('eval')
eval_group.add_argument('--num_eval_processes', type=int, default=5)
eval_group.add_argument('--eval_freq', default=1000, type=int)
eval_group.add_argument('--num_eval_episodes', default=100, type=int)

planner_group = parser.add_argument_group('planner')
planner_group.add_argument('--planner_pos_noise', type=float, default=0)
planner_group.add_argument('--planner_rot_noise', type=float, default=0)

margin_group = parser.add_argument_group('margin')
margin_group.add_argument('--margin', default='l', choices=['ce', 'bce', 'bcel', 'l', 'oril'])
margin_group.add_argument('--margin_l', type=float, default=0.1)
margin_group.add_argument('--margin_weight', type=float, default=0.1)
margin_group.add_argument('--margin_beta', type=float, default=100)

buffer_group = parser.add_argument_group('buffer')
buffer_group.add_argument('--buffer', default='per_expert', choices=['normal', 'per', 'expert', 'per_expert'])
buffer_group.add_argument('--per_eps', type=float, default=1e-6, help='Epsilon parameter for PER')
buffer_group.add_argument('--per_alpha', type=float, default=0.6, help='Alpha parameter for PER')
buffer_group.add_argument('--per_beta', type=float, default=0.4, help='Initial beta parameter for PER')
buffer_group.add_argument('--per_expert_eps', type=float, default=1)
buffer_group.add_argument('--per_td_error', type=str, default='last', choices=['all', 'last'])
buffer_group.add_argument('--batch_size', type=int, default=16)
buffer_group.add_argument('--buffer_size', type=int, default=100000)
buffer_group.add_argument('--fixed_buffer', action='store_true')

logging_group = parser.add_argument_group('logging')
logging_group.add_argument('--log_pre', type=str, default='/tmp')
logging_group.add_argument('--log_sub', type=str, default=None)
logging_group.add_argument('--no_bar', action='store_true')
logging_group.add_argument('--time_limit', type=float, default=10000)
logging_group.add_argument('--load_sub', type=str, default=None)

test_group = parser.add_argument_group('test')
test_group.add_argument('--test', action='store_true')

args = parser.parse_args()
# env
random_orientation = args.random_orientation
env = args.env
num_objects = args.num_objects
max_episode_steps = args.max_episode_steps
action_sequence = args.action_sequence
num_processes = args.num_processes
render = args.render
robot = args.robot


workspace_size = args.workspace_size
if env.find('bumpy') > -1:
    workspace = np.asarray([[0.45-workspace_size/2, 0.45+workspace_size/2],
                            [0-workspace_size/2, 0+workspace_size/2],
                            [0, 0+workspace_size]])
else:
    workspace = np.asarray([[0.5-workspace_size/2, 0.5+workspace_size/2],
                            [0-workspace_size/2, 0+workspace_size/2],
                            [0, 0+workspace_size]])
heightmap_size = args.heightmap_size
patch_size = args.patch_size

if env in ['block_picking', 'random_picking', 'random_float_picking', 'cube_float_picking', 'drawer_opening']:
    num_primitives = 1
else:
    num_primitives = 2

heightmap_resolution = workspace_size/heightmap_size
action_space = [0, heightmap_size]

num_rotations = args.num_rotations
half_rotation = args.half_rotation
if half_rotation:
    rotations = [np.pi / num_rotations * i for i in range(num_rotations)]
else:
    rotations = [(2 * np.pi) / num_rotations * i for i in range(num_rotations)]
in_hand_mode = args.in_hand_mode

######################################################################################
# training
alg = args.alg
if alg == 'dqn_sl_anneal':
    args.sl = True
model = args.model
lr = args.lr
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
sl = args.sl
planner_episode = args.planner_episode

load_model_pre = args.load_model_pre
is_test = args.test
note = args.note
seed = args.seed
perlin = args.perlin
gaussian = args.gaussian

q2_model = args.q2_model

equi_n = args.equi_n

aug = args.aug
aug_type = args.aug_type

# eval
num_eval_processes = args.num_eval_processes
eval_freq = args.eval_freq
num_eval_episodes = args.num_eval_episodes

# pre train
fill_buffer_deconstruct = args.fill_buffer_deconstruct
load_n = args.load_n
expert_aug_n = args.expert_aug_n
expert_aug_d4 = args.expert_aug_d4
pre_train_step = args.pre_train_step

# planner
planner_pos_noise = args.planner_pos_noise
planner_rot_noise = args.planner_rot_noise

# buffer
buffer_type = args.buffer
per_eps = args.per_eps
per_alpha = args.per_alpha
per_beta = args.per_beta
per_expert_eps = args.per_expert_eps
per_td_error = args.per_td_error
batch_size = args.batch_size
buffer_size = args.buffer_size
fixed_buffer = args.fixed_buffer

# margin
margin = args.margin
margin_l = args.margin_l
margin_weight = args.margin_weight
margin_beta = args.margin_beta

# logging
log_pre = args.log_pre
log_sub = args.log_sub
no_bar = args.no_bar
time_limit = args.time_limit
load_sub = args.load_sub
if load_sub == 'None':
    load_sub = None

# z
num_zs = args.num_zs
min_z = args.min_z
max_z = args.max_z

######################################################################################
env_config = {'workspace': workspace, 'max_steps': max_episode_steps, 'obs_size': heightmap_size, 'in_hand_size': patch_size,
              'fast_mode': True,  'action_sequence': action_sequence, 'render': render, 'num_objects': num_objects,
              'random_orientation':random_orientation, 'robot': robot, 'workspace_check': 'point', 'in_hand_mode': in_hand_mode,
              'object_scale_range': (0.6, 0.6), 'hard_reset_freq': 1000, 'physics_mode' : 'fast',}
planner_config = {'pos_noise': planner_pos_noise, 'rot_noise': planner_rot_noise, 'random_orientation':random_orientation, 'half_rotation': half_rotation}

if env in ['block_bin_packing']:
    env_config['object_scale_range'] = (0.8, 0.8)
    env_config['min_object_distance'] = 0.1
    env_config['min_boarder_padding'] = 0.05
if env in ['random_block_picking_clutter']:
    env_config['object_scale_range'] = (0.8, 0.8)
    env_config['min_object_distance'] = 0
    env_config['min_boarder_padding'] = 0.15
    env_config['adjust_gripper_after_lift'] = True
if env in ['bottle_tray', 'box_palletizing', 'bumpy_box_palletizing']:
    env_config['object_scale_range'] = (0.8, 0.8)
    env_config['kuka_adjust_gripper_offset'] = 0.0025
if env in ['covid_test']:
    env_config['object_scale_range'] = (0.55, 0.55)
if seed is not None:
    env_config['seed'] = seed
######################################################################################
hyper_parameters = {}
for key in sorted(vars(args)):
    hyper_parameters[key] = vars(args)[key]

for key in hyper_parameters:
    print('{}: {}'.format(key, hyper_parameters[key]))