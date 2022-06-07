.. BulletArm documentation master file, created by
   sphinx-quickstart on Tue Jan 18 13:15:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/ColinKohler/BulletArm

BulletArm Documentation
============================================

**BulletArm** is a PyBullet based package for robotics maipulation research.

This package contains various simulated robotics environments used for research in the 
Helping Hands lab. The majority of these environments entail a robotic arm armed with a
paralel jaw gripper executing a series of manipulation based tasks.

Package Reference
============================================

The package is structured into two subpackages:

* :doc:`bulletarm <api/bulletarm>` contains the PyBullet environments, tasks, and robots.
* :doc:`bulletarm_baselines <api/bulletarm_baselines>` implements several baseline algorithms alongside logging and plotting tools.

To get started, we provide a number of `tutorials <https://github.com/ColinKohler/BulletArm/tutorials>`_ including an 
`introcutory tutorial <https://github.com/ColinKohler/BulletArm/blob/main/tutorials/run_task.py>`_ demonstrating how to run existing task and 
more advanced `tutorial <https://github.com/ColinKohler/BulletArm/blob/main/tutorials/new_task.py>`_ how to create a new task.

Cite Us
============================================
The development of this package was part of the work done in our ISRR 22 paper. Please,
cite us if you use this code in your own work::

  @misc{https://doi.org/10.48550/arxiv.2205.14292,
    doi = {10.48550/ARXIV.2205.14292},
    url = {https://arxiv.org/abs/2205.14292},
    author = {Wang, Dian and Kohler, Colin and Zhu, Xupeng and Jia, Mingxi and Platt, Robert},
    keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {BulletArm: An Open-Source Robotic Manipulation Benchmark and Learning Framework},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
  }


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
