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

Env Factory
============================================
.. automodule:: helping_hands_rl_envs.env_factory
   :members:

Single Env Runner
============================================
.. autoclass:: helping_hands_rl_envs.runner.SingleRunner
   :members:

Multi Env Runner
============================================
.. autoclass:: helping_hands_rl_envs.runner.MultiRunner
   :members:

Base Env
============================================
.. automodule:: helping_hands_rl_envs.envs.base_env
  :members:

Baselines API
============================================
.. automodule:: baselines.rs
  :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
