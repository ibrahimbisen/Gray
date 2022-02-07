import pybullet as p
import pybullet_data
client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF(r"plane.urdf")
robotId = p.loadURDF(r"urdf/Main Body URDF.urdf", basePosition=[0,0,0.5])
position, orientation = p.getBasePositionAndOrientation(robotId)
while True: 
    p.stepSimulation()