import pybullet as p
import pybullet_data

# Create a client of pybullet physics server
client = p.connect(p.GUI)

p.setGravity(0, 0, -10, physicsClientId=client)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("gray.urdf", basePosition=[0,0,0.5])

position, orientation = p.getBasePositionAndOrientation(robotId)

# Step through simulation (each step is 1/240 of a second)
while True: 
    p.stepSimulation()