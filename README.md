# Helping Hands RL Environments
- [License](https://github.com/ColinKohler/helping_hands_rl_envs/blob/master/LICENSE)

This package contains various simulated robotics environments used for research in the [Helping Hands](https://www2.ccs.neu.edu/research/helpinghands/) lab.
The majority of these environments entail a robotic arm armed with a paralel jaw gripper executing a series of manipulation based tasks. For a full list of 
the tasks currently implemented see below. The core simulator used for most tasks is [PyBullet](https://github.com/bulletphysics/bullet3) but a simple numpy
based simulator is included for quick prototyping. 

## Table of Contents
1. [Requirments](#requirements)
2. [Installation](#install)
3. [Environments](#envs)
4. [Parameters](#parameters)
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
    import numpy as np
    from helping_hands_rl_envs import env_factory
    
    workspace = np.asarray([[0.3, 0.6],
                            [-0.15, 0.15],
                            [0, 0.50]])
    # environment parameters
    env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': True, 'fast_mode': True,
                  'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 4, 'random_orientation': True,
                  'reward_type': 'sparse', 'robot': 'kuka', 'workspace_check': 'point'}
    # planner parameters
    planner_config = {'random_orientation': True}
    # create 1 block_stacking environment
    env = env_factory.createEnvs(1, 'pybullet', 'block_stacking', env_config, planner_config)
    env.reset()
    for i in range(5, -1, -1):
        action = env.getNextAction()
        (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
        input('press enter to continue')
    env.close()
    ```

## Environments
### 2.5D Environments
#### Block Stacking Environments
- **block_picking**: Pick up all blocks.
- **block_stacking**: Pick up blocks and place them in a stack. 
- **block_adjacent**: TODO
- **brick_stacking**: TODO
- **pyramid_stacking**: TODO
- **house_building_1**: Build a house structure with a block stack and a small triangle roof on top.
- **house_building_2**: Build a house structure with two block bases and a large triangle roof.
- **house_building_3**: Build a house structure with two block bases, one brick in the middle, and a large triangle roof.
- **house_building_4**: Build a house structure with two block bases, one brick in the second layer, two blocks in the third layer, and a large triangle roof.
- **house_building_x**: TODO
- **improvise_house_building_2**: Build a house structure with two random shape bases and a large triangle roof.
- **improvise_house_building_3**: Build a house structure with two random shape bases, one brick in the middle, and a large triangle roof.
- **improvise_house_building_discrete**: Build a house structure with 4 random shapes and a large triangle roof. The heights of the random shapes are sampled from two discrete numbers.
- **improvise_house_building_random**: Build a house structure with 4 random shapes and a large triangle roof. The heights of the random shapes are sampled from continuous numbers.
#### Realistic Environments
- **block_bin_packing**: Pack blocks in a box.
- **bottle_tray**: Arrange 6 bottles in a tray.
- **box_palletizing**: Stack boxes on top of a pallet.
- **covid_test**: Supervise 3 covid tests.

### 6D Environments
- **ramp_block_stacking**: Finish block_stacking in the workspace with two ramps
- **ramp_house_building_1**: Finish house_building_1 in the workspace with two ramps
- **ramp_house_building_2**: Finish house_building_2 in the workspace with two ramps
- **ramp_house_building_3**: Finish house_building_3 in the workspace with two ramps
- **ramp_house_building_4**: Finish house_building_4 in the workspace with two ramps
- **ramp_improvise_house_building_2**: Finish improvise_house_building_2 in the workspace with two ramps
- **ramp_improvise_house_building_3**: Finish improvise_house_building_3 in the workspace with two ramps
- **bumpy_house_building_4**: Finish house_building_4 in the workspace with a bumpy surface
- **bumpy_box_palletizing**: Finish bumpy_box_palletizing in the workspace with a bumpy surface

<a name="parameters"></a>
## Parameters

<a name="publications"></a>
## Publications
If you like this package and use it in your own work, please cite this repository.
