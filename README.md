# Helping Hands RL Environments 
This repository holds the environments for the various pick and place tasks we have been working on in the lab. At the moment all 
of the environments are 2.5D top-down grasping tasks where a robot arm has to pick/place object on a table. Moving forward I
expect to extend this to work in 3D with 6-DOF grasping on more complicated tasks. 

# Data Collection
Originally this reposity was designed for online learning but it works equally well for data collection. Although you can use the
master branch as is for data collection, The cpk_refactor branch has a number of additional features, including planners, which
make data collection much easier. This branch will eventually get merged into master once finalized.

# Extending this Repository
If you are simply using this as a base for a very different problem feel free to fork this repository but if you are simply 
extending the existing functionality, such as adding new environments, please considering contributing to the repository! 
In order to keep things clean, please cut a branch for the feature you are working on and submit a pull request when its complete.
If there is additional functionality you think would be nice to have but are unsure how to implement, I would suggest opening a 
issue and we can discuss it there!

# Environments
- **BlockPicking:** Pick up a block on a table. Block size and initial pose are stochastic.
- **BlockStacking:** Pick up blocks and place them in a stack. Block size and initial pose are 
                     stochastic.
- **BrickStacking:** Pick up rectangular blocks and place them in a stack. Block size and initial pose are 
                     stochastic.
- **BlockCylinderStacking:** Pick up disks and place them in a stack. Disk size and initial pose are 
                             stochastic.
- **HouseBuilding1**:
- **HouseBuilding2**:

# Simulators
- **PyBullet Simulator:** Primary simulator.
- **Numpy Simulator:** Less realistic but much fast. Typically used for early prototyping.

# Getting Started
Coming soon! For now you can look in the 'tests' directory to get an idea of how to use things.
