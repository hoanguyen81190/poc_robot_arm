import sys
import os

### In order to be able to import scripts from the SharedScripts directory, we need to append the parent directory to the system path
# Get the grandparent directory of the current script
current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.realpath(current_dir))

# Append the grandparent directory to the system path
sys.path.append(parent_dir)

# In order to control the robot arm using inverse kinematics we need two inputs. 
#   1. the desired position of the end effector (XYZ of the hand, perhaps the average of the 
#  
#   2. the desired orientation of the end effector (Determine this with 
#       the angles between the hand joints, IDs: 16, 18, 20 and 22)


# FOR PID control we will need to read back the robot arm joint angles 
# and use them to calculate the error for the control. 


#from coppeliasim_zmqremoteapi_client import RemoteAPIClient

#client = RemoteAPIClient()
#sim = client.require('sim')


### Using ikpy for inverse kinematics calculations - pip install ikpy

### For controlling the robot arm outside of CoppeliaSim we will use the RTDE interface
#import rtde_receive
#import rtde_control

#rtde_r = rtde_receive.RTDEReceiveInterface("10.213.55.123")
#rtde_c = rtde_control.RTDEControlInterface("10.213.55.123")

from ikpy.chain import Chain
from ikpy.utils import plot
import numpy as np

import matplotlib.pyplot as plt

# Load the URXe URDF file into the kinematic chain
robot_chain = Chain.from_urdf_file(parent_dir+"/urdf/UR16e_final.urdf", active_links_mask=[False,False,True,True,True,True,True,True,False,False])
#robot_chain = Chain.from_urdf_file(parent_dir+"/urdf/UR16e_endEffector.urdf", active_links_mask=[False,True,True,True,True,True,True,False])

print(robot_chain.links)

# Desired 3D position of the end-effector in meters
#target_position = [3, 0.5, 0.4]  # x, y, z coordinates

# Rotation matrix for 45 degrees along the X-axis (in radians)
#theta_x = np.pi / 2  # 45 degrees in radians
#rotation_matrix_x = np.array([
#    [1, 0, 0],
#    [0, np.cos(theta_x), -np.sin(theta_x)],
#    [0, np.sin(theta_x), np.cos(theta_x)]
#])

# Create the full transformation matrix (4x4 matrix)
# The last row is [0, 0, 0, 1], standard in homogeneous coordinates
#transformation_matrix = np.eye(4)
#transformation_matrix[:3, :3] = rotation_matrix_x  # Insert the rotation matrix
#transformation_matrix[:3, 3] = target_position     # Insert the target position

# Robot arm constraints (Might have to put these constraints in the output of the inverse kinematics function)
# These constraints are defined in the urdf file!
# TODO define these constraints in the urdf file
#joint_limits = [
#    (-2 * np.pi, 2 * np.pi),  # Joint 1
#    (-2 * np.pi, 2 * np.pi),  # Joint 2
#    (-2 * np.pi, 2 * np.pi),  # Joint 3
#    (-2 * np.pi, 2 * np.pi),  # Joint 4
#    (-2 * np.pi, 2 * np.pi),  # Joint 5
#    (-2 * np.pi, 2 * np.pi)   # Joint 6
#]


def getJointAnglesFromPose(target_position, target_orientation=np.eye(3), orientation_mode="all", plotFig=False):
    # Compute inverse kinematics solution to get joint angles
    joint_angles = robot_chain.inverse_kinematics(target_position=target_position, max_iter=1000)#,
                                                #target_orientation=target_orientation,orientation_mode=orientation_mode)
    
    #print(robot_chain.links)

    # Plot the robot in its computed configuration
    if plotFig:
        fig, ax = plot.init_3d_figure()
        robot_chain.plot(joint_angles, ax, target=target_position)
        plt.pause(0.1)
        plt.show()

    return joint_angles

def normalize_angle(angle):
    # Normalize angle to the range [pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def clamp_angle(angle):
    # Clamp angle to the range [pi, pi]
    return max(min(angle, np.pi), -np.pi)

def testForwardIKPY(joint_angles, plotFig=False):
    import numpy as np
    from ikpy.chain import Chain

    # Inspect the chain structure
    #print(f"Number of links: {len(robot_chain.links)}")
    #print(f"Joint types: {[link.name for link in robot_chain.links]}")
    for link in robot_chain.links:
        print(f"Link name: {link.name}, Joint type: {link.joint_type}")

    # Define joint angles in radians (example values)
    #joint_angles = [0, np.pi/4, -np.pi/2, np.pi/3, np.pi/6, 0]

    # Compute forward kinematics
    end_effector_frame = robot_chain.forward_kinematics(joint_angles)

    # Plot the robot arm using the joint angles
    fig, ax = plot.init_3d_figure()

    # Plot the robot chain using the calculated joint angles
    robot_chain.plot(joint_angles, ax, target=None)

    # Display the plot
    plt.pause(0.1)

    return end_effector_frame

def convert_to_left_handed(v):   
    return np.array([v[0], v[1], -v[2]])

def convert_mediapipe_to_coppeliasim(v):
    return np.array([v[2], v[0], v[1]])

def map_to_range(value, old_min=0, old_max=1, new_min=-1.3, new_max=1.3):
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

# Alpha determines the weight of the most recent data point
def exponential_smoothing(data, alpha):
    smoothed_value = data[0]  # Initialize with the first data point
    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_value
    return smoothed_value

def deadband_control(target_position, current_position, deadband_threshold = 0.02):
    error = abs(target_position - current_position)
    
    # print(error)

    if max(error) > deadband_threshold: #TODO perhaps this should be the sum of the absolute error instead
        # Apply control if the error exceeds the threshold
        return target_position, True
    else:
        # No control signal applied if within deadband
        return current_position, False