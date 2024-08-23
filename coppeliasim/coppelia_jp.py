from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

import cv2
import mediapipe as mp
import threading, queue
import keyboard
import math
import numpy as np

pose_queue = queue.Queue()

def pose_extraction():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Open the webcam
    # Was capture id 1
    cap = cv2.VideoCapture(0)

    command_frame_interval = 60
    current_frame_count = 0
    while cap.isOpened():
        if keyboard.is_pressed('esc'):  # Check if ESC key is pressed
            print("ESC key pressed. Stopping simulation.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose
        results = pose.process(rgb_frame)

        right_arm_landmarks = [12, 14, 16]
        landmarks_data = []
        # Draw the pose annotation on the image
        if results.pose_landmarks:
            landmarks = [results.pose_landmarks.landmark[idx] for idx in right_arm_landmarks]
            landmarks_data = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks])

            # SEND TO SIMULATION
            if current_frame_count > command_frame_interval:
                current_frame_count -= command_frame_interval
                pose_queue.put(landmarks_data)

            #print(results.pose_landmarks)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            current_frame_count += 1

        # Display the output
        cv2.imshow('Pose Estimation', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def cartesian_to_cylindrical_3d(data):
    x, y, z = data[0], data[1], data[2]
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta, z

prev_shoulder = 0
prev_elbow = 0
prev_wrist = 0

def pose_to_joint_angles(data):
    _, shoulder, z_shoulder = cartesian_to_cylindrical_3d(data[1] - data[0])
    _, elbow, z_elbow = cartesian_to_cylindrical_3d(data[2] - data[1])
    _, wrist, z_wrist = cartesian_to_cylindrical_3d(data[2])

    if data[0][3] < 0.5:
        shoulder = prev_shoulder
    else:
        prev_shoulder = shoulder
    if data[1][3] < 0.5:
        elbow = prev_shoulder
    else:
        prev_elbow = elbow

    elbow = elbow - shoulder
    wrist = wrist - elbow - shoulder

    return shoulder, elbow, wrist

def simulation():
    client = RemoteAPIClient()
    sim = client.require('sim')

    sim.setStepping(True)

    joint_names = [
        'UR10_joint1',
        'UR10_joint2',
        'UR10_joint3',
        'UR10_joint4',
        'UR10_joint5',
        'UR10_joint6'
    ]

    sim.startSimulation()

    joint_handles = []
    for name in joint_names:
        handle = sim.getObjectHandle(name)
        if handle != -1:
            joint_handles.append(handle)
            print(f'Handle for {name} retrieved successfully')
        else:
            print(f'Failed to get handle for {name}')

    # Continuously step the coppelia simulation
    simulation_time = 0
    while (True):
        if keyboard.is_pressed('esc'):  # Check if ESC key is pressed
            print("ESC key pressed. Stopping simulation.")
            break
        
        # Update the pose location if available
        base = 0
        if not pose_queue.empty():
            landmarks_data = pose_queue.get()
            shoulder, elbow, wrist = pose_to_joint_angles(landmarks_data)
            #Joint positions for the UR10 robot
            # 0,0,0,0 is the home position which leaves the arm pointing straight up
            # 0 is the base and rotates around the up axis
            # 1 is the first joint and corresponds to the shoulder and rotates around the x or y axis
            # 2 is the second joint and corresponds to the elbow and rotates around the x or y axis
            # 3 is the third joint and corresponds to the wrist and rotates around the x or y axis
            joint_positions = [base, shoulder, elbow, wrist, 0, 0]  # Example positions in radians

            for i, handle in enumerate(joint_handles):
                sim.setJointTargetPosition(handle, joint_positions[i])

        # Step the simulation
        #simulation_time = 0
        #while simulation_time < 5:  # Run for 5 seconds
        print(f'Simulation time: {simulation_time:.2f} [s]')
        sim.step()
        simulation_time = sim.getSimulationTime()
        time.sleep(0.1)  # Sleep to avoid excessive CPU usage

    # Stop the simulation
    sim.stopSimulation()

if __name__ == '__main__':
    pose_extraction_thread = threading.Thread(target=pose_extraction, )
    pose_extraction_thread.start()

    simulation_thread = threading.Thread(target=simulation, )
    simulation_thread.start()