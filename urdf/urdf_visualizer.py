import pybullet as p
import pybullet_data

import pathlib
import os

# Start PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


# Load your URDF model
path = os.path.abspath("./UR16e_gpt.urdf")
print(path)
robot_id = p.loadURDF(path)

# Run the simulation
while True:
    p.stepSimulation()