# BulletArm
- [License](https://github.com/ColinKohler/BulletArm/blob/main/LICENSE)

This package contains various simulated robotics environments used for research in the [Helping Hands](https://www2.ccs.neu.edu/research/helpinghands/) lab.
The majority of these environments entail a robotic arm armed with a paralel jaw gripper executing a series of manipulation based tasks. For a full list of 
the tasks currently implemented see below. The core simulator used for most tasks is [PyBullet](https://github.com/bulletphysics/bullet3) but a simple numpy
based simulator is included for quick prototyping. 

## Table of Contents
1. [Requirments](#requirments)
2. [Installation](#install)
3. [Environments](#envs)
4. [Parameters](#parameters)
1. [Benchmarks](#benchmarks)
5. [Publications](#publications)

<a name="requirments"></a>
## Requirments

<a name="install"></a>
## Install
1. Install Python 3.7
2. Clone this repo
    ```
    git clone https://github.com/ColinKohler/helping_hands_rl_envs.git
    cd helping_hands_rl_envs
    ```
3. Install dependencies
    ```
    pip install -r requirements.txt 
    ```
4. Install this package
    ```
    pip install .
    ```
5. Run the block stacking demo
    ```python
    from bulletarm import env_factory
    # environment parameters
    env_config = {'render': True}
    # create 1 block_stacking environment
    env = env_factory.createEnvs(1, 'block_stacking', env_config)
    env.reset()
    for i in range(5, -1, -1):
        action = env.getNextAction()
        (states_, in_hands_, obs_), rewards, dones = env.step(action)
        input('press enter to continue')
    env.close()
    ```
<a name="envs"></a>
## Environments
### Open-Loop Environments
<img src="/img/open_envs.png" width="640">

- **block_picking**: The robot needs to pick up all N cubic blocks. The number of blocks N in this environments is configurable. 
- **block_stacking**: (a) The robot needs to stack all N cubic blocks. The number of blocks N in this environments is configurable. 
- **house_building_1**: (b) The robot needs to first stack N-1 cubic blocks then place a triangle block on top of the stack. The number of blocks N in this environments is configurable. 
- **house_building_2**: (c) The robot needs to first place two cubic blocks adjacent to each other, then place a roof on top.
- **house_building_3**: (d) The robot needs to: 1. place two cubic blocks adjacent to each other; 2. put a cuboid on top of the two cubic blocks; 3. put a roof on top of the cuboid.
- **house_building_4**: (e) The robot needs to: 1. place two cubic blocks adjacent to each other; 2. put a cuboid on top of the two cubic blocks; 3. put another two cubic blocks on top of the cuboid; 4. put a roof on top of the structure.
- **improvise_house_building_2**: (f) Similar task as House Building 2, but the fixed cubic blocks are replaced with random shape blocks.
- **improvise_house_building_3**: (g) Similar task as House Building 3, but the fixed cubic blocks are replaced with random shape blocks.
- **block_bin_packing**: (h) The robot needs to pack the N blocks in the workspace inside a bin. The number of blocks N in this environments is configurable. 
- **bottle_tray**: (i) The robot needs to arrange six bottles in the tray.
- **box_palletizing**: (j) The robot needs to palletize N boxes on top of a pallet. The number of boxes N in this environment is configurable (6, 12, or 18). This environments is first proposed in~\cite{transporter}.
- **covid_test**: (k) The robot needs to supervise three covid tests and gather the test tubes.
- **object_grasping**: (l) The robot needs to pick up an object in a cluttered scene containing N random objects. The number of objects N in this environment is configurable.

### Open-Loop 6D Environments
<img src="/img/open_6d_envs.png" width="480">

- **ramp_block_stacking**: (a) Finish block_stacking in the workspace with two ramps
- **ramp_house_building_1**: (b) Finish house_building_1 in the workspace with two ramps
- **ramp_house_building_2**: (c) Finish house_building_2 in the workspace with two ramps
- **ramp_house_building_3**: (d) Finish house_building_3 in the workspace with two ramps
- **ramp_house_building_4**: (e) Finish house_building_4 in the workspace with two ramps
- **ramp_improvise_house_building_2**: (f) Finish improvise_house_building_2 in the workspace with two ramps
- **ramp_improvise_house_building_3**: (g) Finish improvise_house_building_3 in the workspace with two ramps
- **bumpy_house_building_4**: (h) Finish house_building_4 in the workspace with a bumpy surface
- **bumpy_box_palletizing**: (i) Finish box_palletizing in the workspace with a bumpy surface

### Close-Loop Environments
- **close_loop_block_reaching**: The robot needs to place the gripper close to a cubic block.
- **close_loop_block_picking**: The robot needs to pick up a cubic block.
- **close_loop_block_pushing**: The robot needs to push the block into a goal area.
- **close_loop_block_pulling**: The robot needs to pull one of the two blocks to make contact with the other block.
- **close_loop_block_in_bowl**: The robot needs to pick up a block and place it inside a bowl.
- **close_loop_block_stacking**: The robot needs to stack N cubic blocks. The number of blocks N in this environment is configurable. 
- **close_loop_house_building_1**: The robot needs to stack N-1 cubic blocks then place a triangle block on top of the stack. The number of blocks N in this environments is configurable.
- **close_loop_block_picking_corner**: The robot needs to slide the block from the corner and then pick it up.
- **close_loop_drawer_opening**: The robot needs to pull the handle of the drawer to open it.
- **close_loop_clutter_picking**:  The robot needs to pick up an object in a cluttered scene containing N random objects. The number of objects N in this environment is configurable.

<a name="parameters"></a>
## Parameters

<a name="benchmarks"></a>
## Benchmarks
### Open-Loop Benchmarks
#### Prerequisite
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. (Optional, required for 6D benchmark) Install [CuPy](https://github.com/cupy/cupy)
1. Install other required packages
    ```
    pip install -r baseline_requirement.txt
    ```
1. Goto the baseline directory
    ```
    cd helping_hands_rl_envs/helping_hands_rl_baselines/fc_dqn/scripts
    ```
#### Open-Loop 3D Benchmark
```
python main.py --algorithm=[algorithm] --architecture=[architecture] --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_asr` (recommended), `cnn_asr`, `equi_fcn`, `cnn_fcn`, `rot_fcn`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.
#### Open-Loop 2D Benchmark
```
python main.py  --algorithm=[algorithm] --architecture=[architecture] --action_sequence=xyp --random_orientation=f --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_fcn` (recommended), `cnn_fcn`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.
#### Open-Loop 6D Benchmark
```
python main.py  --algorithm=[algorithm] --architecture=[architecture] --action_sequence=xyzrrrp --patch_size=[patch_size] --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_deictic_asr` (recommended), `cnn_asr`
- Set `[patch_size]` to be `40` (required for `bumpy_box_palletizing` environment) or `24`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.

#### Additional Training Arguments
See [bulletarm_baselines/fc_dqn/utils/parameters.py](bulletarm_baselines/fc_dqn/utils/parameters.py)

### Close-Loop Benchmarks
#### Prerequisite
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. (Optional, required for 6D benchmark) Install [CuPy](https://github.com/cupy/cupy)
1. Install other required packages
    ```
    pip install -r baseline_requirement.txt
    ```
1. Goto the baseline directory
    ```
    cd helping_hands_rl_envs/helping_hands_rl_baselines/equi_rl/scripts
    ```
#### Close-Loop 4D Benchmark
```
python main.py --algorithm=[algorithm] --env=[env]
```
- Select `[algorithm]` from: `sac`, `sacfd`, `equi_sac`, `equi_sacfd`, `ferm_sac`, `ferm_sacfd`, `rad_sac`, `rad_sacfd`, `drq_sac`, `drq_sacfd`
- To use PER and data augmentation buffer, add `--buffer=per_expert_aug`
#### Close-Loop 3D Benchmark
```
python main.py --algorithm=[algorithm] --action_sequence=pxyz --random_orientation=f --env=[env]
```
- Select `[algorithm]` from: `sac`, `sacfd`, `equi_sac`, `equi_sacfd`, `ferm_sac`, `ferm_sacfd`, `rad_sac`, `rad_sacfd`, `drq_sac`, `drq_sacfd`

#### Additional Training Arguments
See [bulletarm_baselines/equi_rl/utils/parameters.py](bulletarm_baselines/equi_rl/utils/parameters.py)

<a name="publications"></a>
## Publications
If you like this package and use it in your own work, please cite this repository.
