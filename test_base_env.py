import pybullet as pb
import pybullet_data
import numpy as np
from pybullet_toolkit.robots.ur5_rg2 import UR5_RG2

client = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setTimeStep(1./240.)
pb.setGravity(0, 0, -10)

table_id = pb.loadURDF('plane.urdf', [0,0,0])
table_id = pb.loadURDF('cube_small.urdf', [0.5,0,0])

ur5 = UR5_RG2()
ur5.reset()
print(ur5.grasp(np.array([0.5, 0.0, 0.23]), 0.05, dynamic=True))

input('end')
