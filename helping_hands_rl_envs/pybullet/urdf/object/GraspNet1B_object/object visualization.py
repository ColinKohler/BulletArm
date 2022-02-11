import pybullet as p
import time
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

rot = (0, 0, 0, 1)
obj_filepath = './003/convex.obj'
obj_id = '0000'
obj_filepath = './' + obj_id + '/textured.obj'
obj_filepath0 = './' + obj_id + '/convex.obj'
# obj_filepath = './' + obj_id + '/textured.obj'
# obj_filepath0 = './' + obj_id + '/nontextured_simplified.ply'
color = np.random.uniform(0.6, 1, (4,))
color[-1] = 1
obj_visual = p.createVisualShape(p.GEOM_MESH, fileName=obj_filepath0,
                                 meshScale=[1, 1, 1])
# obj_visual = None
obj_collision = p.createCollisionShape(p.GEOM_MESH, fileName=obj_filepath, meshScale=[1, 1, 1])

# pos_in_air = pos.copy()
# pos_in_air[2] += 1.5
object_id = p.createMultiBody(baseMass=0.5,
                              baseCollisionShapeIndex=obj_collision,
                              baseVisualShapeIndex=obj_visual,
                              basePosition=[0, 0, 0.5],
                              baseOrientation=rot)

# cubeStartPos = [0,0,1]
# cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()
