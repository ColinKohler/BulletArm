..

Getting Started
================

Installation
--------------

.. code-block:: bash
  
  # Clone the BulletArm Repo
  git clone https://github.com/ColinKohler/BulletArm.git && cd BulletArm

  # Install dependencies
  pip install -r requirements.txt

  # Install BulletArm either using Pip 
  pip install .
  # OR by adding it to your PYTHONPATH 
  export PYTHONPATH=/path/to/BulletArm/:$PYTHONPATH


Block Stacking Demo
--------------------
In order to test your installation, we recommend running the block stacking demo to ensure everything is in working order.

.. code-block:: bash
  
  python tutorials/block_stacking_demo.py

Below we go over the code within the demo and briefly describe the important details.

.. code-block:: python
  :linenos:

  # The env_factory provides the entry point to BulletArm
  from bulletarm import env_factory
  
  def runDemo():
    env_config = {'render': True}
    # The env_factory creates the desired number of PyBullet simulations to run in 
    # parallel. The task that is created depends on the environment name and the 
    # task config passed as input.
    env = env_factory.createEnvs(1, 'block_stacking', env_config)
 
    # Start the task by resetting the simulation environment. 
    obs = env.reset()
    done = False
    while not done:
      # We get the next action using the planner associated with the block stacking 
      # task and execute it.
      action = env.getNextAction()
      obs, reward, done = env.step(action)
    env.close()

Tutorials
------------
We provide a number of `tutorials <https://github.com/ColinKohler/BulletArm/tree/main/tutorials>`_ including an 
`introcutory tutorial <https://github.com/ColinKohler/BulletArm/blob/main/tutorials/train_dummy_agent.py>`_ demonstrating how to collect data
for training of a RL agent. Examples on how to extend PyBullet for either 
`creating new tasks <https://github.com/ColinKohler/BulletArm/blob/main/tutorials/creating_a_new_task.py>`_ or
`creating new robots <https://github.com/ColinKohler/BulletArm/blob/main/tutorials/creating_a_new_robot.py>`_ are also included.
