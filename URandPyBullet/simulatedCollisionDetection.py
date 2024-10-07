import numpy as np
import pybullet as p
import pybullet_data
import time
from Utils.utils import clamp_angle, normalize_angle

class simulationCollisionDetection:
    def __init__(self):
        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)  # Connect to the GUI for visualization
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For loading plane and robot URDF
        p.setGravity(0, 0, 0)  # Set gravity in the negative z direction
        p.setTimeStep(0.01)  # Time step for simulation
        p.setPhysicsEngineParameter(numSolverIterations=10)  # Increase solver iterations

        # Load your robot in PyBullet
        p.loadURDF("plane.urdf")
        self.end_effector_link_index = 6  
        self.robot_id = p.loadURDF("urdf/ur16e_final.urdf", basePosition=[0, 0, 0.01], useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot_id)
        for joint_index in range(self.num_joints):
            p.changeDynamics(self.robot_id, joint_index+1, jointDamping=0.1)

    def __del__(self):
        p.disconnect()

    def is_at_target_position(self, target_angles, tolerance=0.005):
        current_angles = []
        for i in range(len(target_angles)):
            joint_state = p.getJointState(self.robot_id, i+1)
            current_angles.append(joint_state[0])  # joint_state[0] is the current position
        current_angles = np.array(current_angles)
        error = np.abs(current_angles - target_angles)
        return np.all(error < tolerance)

    # Function to get current joint angles
    def get_joint_angles(self):
        joint_angles = []
        num_joints = p.getNumJoints(self.robot_id)  # Get number of joints in the robot
        for joint_index in range(num_joints):
            joint_state = p.getJointState(self.robot_id, joint_index)
            joint_angles.append(joint_state[0])  # Append the joint angle (position)
        return joint_angles

    # Function to get the position of the end effector
    def get_end_effector_position(self):
        # Get the link state of the end effector
        link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)

        # You can print or use the orientation values
        print("End effector position:", link_state[4])
        print("End effector orientation (quaternion):", link_state[5])

        return link_state[0], link_state[5]  # Returns the position as (x, y, z)

    def simulate_path_to_target(self, start_joint_angles, start_position, start_orientation, target_position=[0,0,1.3], num_segments=10):
        ### Set the joint angles of the robot arm to mimick the real robot arm before doing the simulation
        # Loop through each joint and set the joint angle
        for joint_index, angle in enumerate(start_joint_angles):
            p.resetJointState(self.robot_id, joint_index+1, angle)

        collision_detected = False
        #start_position, start_orientation = self.get_end_effector_position()
        #print("Start End Effector Position:", start_position)
        #start_joint_angles = self.get_joint_angles()

        # For storing the intermediate joint angles that IK comes up with
        intermediate_joint_angles = np.zeroes((num_segments,self.num_joints))
        # Create the segments to move from start position to the target position
        intermediate_positions = [start_position + (target_position - start_position) * (i / num_segments) for i in range(num_segments + 1)]
        # Might want to do something similar for the orientation of the end effector, currently just passing the same orientation as at the start

        

        # Movement Loop
        for i, pos in enumerate(intermediate_positions):
            # Get the target pose for the end effector
            target_pose = np.eye(4)
            target_pose[:3, 3] = pos  # Set the target position

            joint_angles = p.calculateInverseKinematics(self.robot_id, self.end_effector_link_index, pos, start_orientation)
            joint_angles = np.asarray([clamp_angle(normalize_angle(angle)) for angle in joint_angles])

            intermediate_joint_angles[i] = joint_angles

            # Set the robot's joint angles in PyBullet
            for y in range(len(joint_angles)):
                p.setJointMotorControl2(self.robot_id, y+1, p.POSITION_CONTROL, targetPosition=joint_angles[y])

            # Step the simulation until the robot reaches the target position
            while not self.is_at_target_position(joint_angles):   
                # Example: Enforce a minimum Z height for all parts of the robot
                for joint_index in range(self.num_joints):
                    # Get link state (position of the center of mass of the link)
                    link_state = p.getLinkState(self.robot_id, joint_index+1)
                    link_z_position = link_state[0][2]  # Get z position
                    
                    # Check if z-coordinate is less than zero
                    if link_z_position < 0:
                        print(f"Joint {joint_index+1} is below the ground!")
                        # Take action: e.g., adjust joint angles or ignore this solution
                
                p.stepSimulation()
                time.sleep(0.01)

            # Visualize the end-effector position
            end_effector_pos, _ = p.getLinkState(self.robot_id, self.end_effector_link_index)[:2]
            
            # Draw the end effector position in the simulation
            p.addUserDebugLine(end_effector_pos, end_effector_pos + np.array([0, 0, 0.1]), lineColorRGB=[1, 0, 0], lifeTime=0.1)

            # Check for self-collision
            collision_info = p.getContactPoints(bodyA=self.robot_id)
            if collision_info:
                print("Self-collision detected!")
                collision_detected = True
                break

        return collision_detected, intermediate_positions, intermediate_joint_angles