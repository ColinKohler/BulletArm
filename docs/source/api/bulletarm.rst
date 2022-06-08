.. automodule:: bulletarm
  :no-members:

.. |check| unicode:: U+2611 .. CHECKMARK 
.. |nocheck| unicode:: U+2610 .. NO_CHECKMARK 

bulletarm
==========

This is the main package for the BulletArm library. It contains the PyBullet simulation environments, robots, and 
scene. Tasks are defined within environments using these tools. 

Tasks
-----------------
Our set of tasks are seperated into two categories based on the action space: open-loop control and closed-loop control.

.. list-table::
  :widths: 25 25 25
  :header-rows: 1
  :align: center

  * - Task
    - Open-Loop Control
    - Closed-Loop Control
  * - :ref:`Block Reaching` 
    - |nocheck| 
    - |check| 
  * - :ref:`Block Pushing` 
    - |nocheck| 
    - |check| 
  * - :ref:`Block Pulling` 
    - |nocheck| 
    - |check| 
  * - :ref:`Block Picking` 
    - |check| 
    - |check| 
  * - :ref:`Block Stacking` 
    - |check| 
    - |check| 
  * - :ref:`Block In Bowl` 
    - |nocheck| 
    - |check| 
  * - :ref:`House Building 1` 
    - |check| 
    - |check| 
  * - :ref:`House Building 2` 
    - |check| 
    - |nocheck| 
  * - :ref:`House Building 3` 
    - |check| 
    - |nocheck| 
  * - :ref:`House Building 4` 
    - |check| 
    - |nocheck| 
  * - :ref:`Improvise House Building 2` 
    - |check| 
    - |nocheck| 
  * - :ref:`Improvise House Building 3` 
    - |check| 
    - |nocheck| 
  * - :ref:`Bin Packing` 
    - |check| 
    - |nocheck| 
  * - :ref:`Bottle Arrangement` 
    - |check| 
    - |nocheck| 
  * - :ref:`Box Palletizing` 
    - |check| 
    - |nocheck| 
  * - :ref:`Covid Test`
    - |check| 
    - |nocheck| 
  * - :ref:`Corner Picking`
    - |nocheck| 
    - |check| 
  * - :ref:`Drawer Opening`
    - |nocheck| 
    - |check| 
  * - :ref:`Object Grasping`
    - |check| 
    - |check| 

EnvFactory & EnvRunner
-----------------------
Blah Blah Blah


