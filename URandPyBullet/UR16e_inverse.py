import sys
import os

### In order to be able to import scripts from the SharedScripts directory, we need to append the parent directory to the system path
# Get the grandparent directory of the current script
current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.realpath(current_dir))

# Append the grandparent directory to the system path
sys.path.append(parent_dir)

import time
import cv2
import mediapipe as mp
import threading, queue
import keyboard
import numpy as np
import rtde_receive
import rtde_control

from URandPyBullet.simulatedCollisionDetection import simulationCollisionDetection
from Utils.utils import deadband_control, map_to_range
from Utils import ExponentialSmoother

running = True
pose_queue = queue.Queue(1)

# Extract the arm joints by default
def pose_extraction(desired_landmark_ids=[12, 14, 16]):
    global running
    # print("Func Inputs: ", desired_landmark_ids)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Open the webcam
    # Was capture id 1
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose
        results = pose.process(rgb_frame)

        right_arm_landmarks = desired_landmark_ids
        landmarks_data = []
        # Draw the pose annotation on the image
        if results.pose_landmarks:
            landmarks = [results.pose_landmarks.landmark[idx] for idx in right_arm_landmarks]
            landmarks_data = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks])

            # SEND TO SIMULATION
            if(pose_queue.empty()):
                pose_queue.put(landmarks_data)

            #print(results.pose_landmarks)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Display the output
        cv2.imshow('Pose Estimation', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def process_landmarks_to_target_position(landmarks_data, max_reach, smoother):
    landmarks_target_position = np.mean(landmarks_data,axis=0)[0:3]

    # Swap axis so it makes sense in terms of controlling the robot arm
    re_mapped = landmarks_target_position.copy()
    re_mapped[0] = -landmarks_target_position[2]
    re_mapped[1] = landmarks_target_position[0]
    re_mapped[2] = landmarks_target_position[1]

    # Map the input ranges so they are more meaningful for cotrolling the robot arm
    re_mapped[1] = map_to_range(re_mapped[1],0,1,-max_reach,max_reach)
    re_mapped[2] = map_to_range(1-re_mapped[2],0,1,0,max_reach) 

    # Apply exponential smoothing 
    target_position = np.asarray([0.1,0.1,target_position[2]]) #TODO update later when done testing
    smoothed_pos = smoother.update(target_position)

    # Clamp the target position to the maximum reach of the robot arm
    target_position_in = np.clip(smoothed_pos, -max_reach, max_reach)
    # Ensure the target position is above the ground
    target_position_in[2] = 0 if target_position_in[2] < 0 else target_position_in[2]
    return target_position

#TODO Implement this function
def get_joint_angles_and_end_effector_pose_from_robot():
    angles = np.zeros(6)
    end_effector_pos = np.zeros(3)
    end_effector_orientation = np.zeros(4)
    return angles, end_effector_pos, end_effector_orientation

def simulation():
    global running

    simCollider = simulationCollisionDetection()

    rtde_r = rtde_receive.RTDEReceiveInterface("10.213.55.123")
    rtde_c = rtde_control.RTDEControlInterface("10.213.55.123")
    
    ### Define robot arm contraints
    max_reach = 0.9

    ### Initialize exponential smoothers for the target position and joints
    alpha = 0.2
    smoother = ExponentialSmoother.ExponentialSmoother(alpha)
    #smoother_joint = ExponentialSmoother.ExponentialSmoother(alpha)

    ### Deadband thresholding for the target position
    bUseDeadbandThresholding = True
    prevDeadbandPos = np.asarray([0,0,0])

    # Continuously step the coppelia simulation
    while (running):
        # Update the pose location if available 
        if not pose_queue.empty():
            # Process the landmarks data to get the target position
            landmarks_data = pose_queue.get_nowait()
            target_position = process_landmarks_to_target_position(landmarks_data, max_reach, smoother)
            # Only update the target position if the deadband thresholding is not applied to avoid unnecessary commands
            if bUseDeadbandThresholding:
                target_position, applied = deadband_control(target_position, prevDeadbandPos, deadband_threshold=0.05)
                if applied:
                    prevDeadbandPos = target_position

            # Get the current joint angles and position and orientation of the end effector from the actual robot
            start_joint_angles, end_effector_pos, end_effector_orientation = get_joint_angles_and_end_effector_pose_from_robot()

            # Simulate the target position in pybullet to ensure that there will be no self collisions before executing on the actual robot
            collisionDetected, intermidiatePositions, intermediate_joint_angles = simCollider.simulate_path_to_target(start_joint_angles, end_effector_pos, end_effector_orientation, target_position=[0,0,1.3], num_segments=10)

            if not collisionDetected:
                for angles in intermediate_joint_angles:
                    # Send the joint angles to the physical robot arm
                    speed = 1.05/4
                    acceleration = 1.4/4
                    asyncCommand = False # If False blocks the thread until the movement is done, otherwise it is non-blocking
                    rtde_c.moveJ(angles, speed, acceleration, asyncCommand)
            else:
                print("Collision detected, not using the target position this frame")
        time.sleep(0.01)  

def check_for_esc():
    global running
    keyboard.wait('esc')  # Wait until the Esc key is pressed
    print("Esc pressed, stopping...")
    running = False       # Set the stop flag

if __name__ == '__main__':
    landmark_ids = [16,18,20]
    pose_extraction_thread = threading.Thread(target=pose_extraction, args=(landmark_ids,))
    pose_extraction_thread.start()

    simulation_thread = threading.Thread(target=simulation, )
    simulation_thread.start()
    
    check_for_esc()
    
    simulation_thread.join()
    pose_extraction_thread.join()