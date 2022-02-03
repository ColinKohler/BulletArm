.. HelpingHandsEnvs documentation master file, created by
   sphinx-quickstart on Tue Jan 18 13:15:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HelpingHands RL Envs
============================================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

This package contains various simulated robotics environments used for research in the 
Helping Hands lab. The majority of these environments entail a robotic arm armed with a
paralel jaw gripper executing a series of manipulation based tasks. For a full list of 
the tasks currently implemented see below. The physics simulator used is PyBullet.

Initialization and Running
============================================
Creating a environment instance is done through the use of the env_factory. Depending
on the number of environments you wish to run in parallel, either a SingleRunner or a 
MultiRunner will be returned which you can interact with.

Env Factory
--------------------------------------------
.. automodule:: helping_hands_rl_envs.env_factory
   :members:

Single Env Runner
--------------------------------------------
.. autoclass:: helping_hands_rl_envs.runner.SingleRunner
   :members:

Multi Env Runner
--------------------------------------------
.. autoclass:: helping_hands_rl_envs.runner.MultiRunner
   :members:

Environments
============================================
A number of different environments are aleady implemented and can found in the list
below. To create your own environment, subclass the base environment and add the new
environment to the list of avaliable environments in the env_fn file.

Envs
--------------------------------------------
* Env 1
* Env 2
* ...

Base Env
--------------------------------------------
.. automodule:: helping_hands_rl_envs.envs.base_env
  :members:

PyBullet Simulator
============================================
The physics simulator used for our domains is PyBullet. We include a number of robotic 
manipulators for use. There are also numerous objects and environment details which can 
be loaded into any enviornment.

Robots
--------------------------------------------
The currently avaliable robots are:

* Kuka
* Floating Kuka Gripper
* UR5 w/Prismatic Gripper
* UR5 w/Robotiq 85-2f Gripper

Additional robots can be added by sub-classing the Robot Base class detailed below. Note:
Robot base contains a number of abstract functions which are specific to the robot being used
and therefor must be implemetned on any new robots.

Robot Base
--------------------------------------------
.. automodule:: helping_hands_rl_envs.pybullet.robots.robot_base
  :members:

Objects
--------------------------------------------
The currently avaliable objects are:

* Bottle
* Bowl
* Box
* Brick
* Cube
* Cup
* Cylinder
* Flat Block
* Pallet
* Plate
* Roof
* Spoon
* Swab
* Teapot Base
* Teapot Lid
* Test Tube
* Triangle

Additional objects can be added by sub-classing the object base class detailed below. The urdf
for the new object must be added to the urdf directory as well.

Object Base
--------------------------------------------
.. automodule:: helping_hands_rl_envs.pybullet.objects.pybullet_object
  :members:

Equipment
--------------------------------------------
The currently avaliable equipment is:

* Blanket
* Cabinet
* Container Box
* Corner
* Drawer
* Drawer Handle
* Drawer w/Rack
* Rack
* Shelf
* Tray

Additional equipment can be added by sub-classing the equipment base class detailed below. The urdf
for the new equipment must be added to the urdf directory as well.

Equipment Base
--------------------------------------------
.. automodule:: helping_hands_rl_envs.pybullet.objects.pybullet_equipment
  :members:

Baselines 
============================================
We include a number of baselines to allow for easier testing and benchmarking of new 
algorithms. Additional baselines will be added over time but pull requests are encouraged
and apperciated.

There are both model-free and model-based methods available, see below for a complete list.

Model-Free
--------------------------------------------
* FC_DQN: Fully convolutional DQN 

Model-Based
--------------------------------------------
* RS: Random sampling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
