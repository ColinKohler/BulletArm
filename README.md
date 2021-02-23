# Helping Hands RL Environments 
This repository holds the environments for the various pick and place tasks we have been working on in the lab. 
The environments include a set of 2.5D top-down tasks where a robot arm has to pick/place object on a table, and a set of 6D tasks where a robot has to pick/place objects initialized on two ramps

## Getting Started
1. Install Python 3.7
1. Clone this repo
    ```
    git clone https://github.com/ColinKohler/helping_hands_rl_envs.git
    cd helping_hands_rl_envs
    ```
1. Install dependencies
    ```
    pip install -r requirements.txt 
    ```
1. Install this package
    ```
    pip install .
    ```
1. Run the block stacking demo
    ```python
    import numpy as np
    from helping_hands_rl_envs import env_factory
    
    workspace = np.asarray([[0.3, 0.6],
                            [-0.15, 0.15],
                            [0, 0.50]])
    env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': True, 'fast_mode': True,
                  'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 4, 'random_orientation': True,
                  'reward_type': 'sparse', 'robot': 'kuka', 'workspace_check': 'point'}
    
    planner_config = {'random_orientation': True}
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

### 6D Environments
- **ramp_block_stacking**: Finish block_stacking in the workspace with two ramps
- **ramp_house_building_1**: Finish house_building_1 in the workspace with two ramps
- **ramp_house_building_2**: Finish house_building_2 in the workspace with two ramps
- **ramp_house_building_3**: Finish house_building_3 in the workspace with two ramps
- **ramp_house_building_4**: Finish house_building_4 in the workspace with two ramps
- **ramp_improvise_house_building_2**: Finish improvise_house_building_2 in the workspace with two ramps
- **ramp_improvise_house_building_3**: Finish improvise_house_building_3 in the workspace with two ramps

## Parameter List
TODO

## Data Collection
Originally this reposity was designed for online learning but it works equally well for data collection. Although you can use the
master branch as is for data collection, The cpk_refactor branch has a number of additional features, including planners, which
make data collection much easier. This branch will eventually get merged into master once finalized.

## Extending this Repository
If you are simply using this as a base for a very different problem feel free to fork this repository but if you are simply 
extending the existing functionality, such as adding new environments, please considering contributing to the repository! 
In order to keep things clean, please cut a branch for the feature you are working on and submit a pull request when its complete.
If there is additional functionality you think would be nice to have but are unsure how to implement, I would suggest opening a 
issue and we can discuss it there!

