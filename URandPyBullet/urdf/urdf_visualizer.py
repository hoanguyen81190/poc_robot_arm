import sys
import os

### In order to be able to import scripts from the SharedScripts directory, we need to append the parent directory to the system path
# Get the grandparent directory of the current script
current_dir = os.getcwd()
parent_dir = os.path.realpath(current_dir)

print("PARENT DIR: ", parent_dir)

# Append the grandparent directory to the system path
sys.path.append(parent_dir)

import pybullet as p
import pybullet_data

# Start PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load your URDF model
print(parent_dir)
path = os.path.abspath(parent_dir+"/urdf/ur16e_final.urdf")
print(path)
robot_id = p.loadURDF(path)

num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    print(info)

# Run the simulation
while True:
    p.stepSimulation()