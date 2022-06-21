BulletArm Overview 
============================================

**BulletArm** is a PyBullet based benchmark and learning-environment for robotic manipulation research.

This package is mainly comprised of two sub-packages: the :ref:`BulletArm Simulator` and 
the :ref:`BulletArm Baselines`. The simulator provides numerous tools to enable
rapid development of robotic manipulation environments for use in robotic learning. It provides a number 
of environemnts, robots, planners, and tasks out-of-the-box. Additionally, BulletArm was designed with
an emphasis on extensibility, allowing users to easily create new enviornments, tasks, and robots as needed.
We aim to maintain and improve this package adding new features over time and hope to encourage the community
to assist with this by adding any interesting tasks they might design.

Through the combination of these tasks with the provided baselines, we present a comprehensive benchmark for
use in either open-loop or closed-loop control. This benchmark consists of 19 different manipulation tasks 
of varying difficuly, ranging from simple reaching and picking tasks to more realistic tasks such as bin packing
and pallet stacking. These tasks can be seen below (for more information see :ref:`Benchmark Tasks`).
We provide state-of-the-art baseline algorithms allowing users to compare their algorithms to our previous work 
on any of the existing tasks or novel tasks which they create.

.. figure:: ../images/tasks.png
  :figwidth: 600px
  :align: center

  Benchmark Tasks.
