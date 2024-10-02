from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

import cv2
import mediapipe as mp
import threading, queue
import keyboard
import math
import numpy as np
import signal 
import time
    
import rtde_receive
import rtde_control

pose_queue = queue.Queue()

stop_flag = False

def pose_extraction():
    global stop_flag
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Open the webcam
    # Was capture id 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    command_frame_interval = 1
    current_frame_count = 0
    while cap.isOpened() and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("Stopping because empty camera frame.")
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
            #if current_frame_count > command_frame_interval:
            #    current_frame_count -= command_frame_interval
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

def vector_between_points(p1, p2):
    return np.array([p2[i] - p1[i] for i in range(3)])

def angle_between_vectors(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
    return np.arccos(cos_theta)  # in radians

def pose_to_joint_angles(data):
    global prev_shoulder, prev_elbow, prev_wrist
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

    return abs(math.pi/2 - (shoulder % math.pi)), elbow % math.pi, abs(math.pi/2 - (wrist % math.pi))

def convert_to_left_handed(v):
    """ Converts a vector from a right-handed to a left-handed coordinate system by inverting the Z component """
    return np.array([v[0], v[1], -v[2]])

def robot_control():
    global stop_flag

    rtde_r = rtde_receive.RTDEReceiveInterface("10.213.55.123")
    rtde_c = rtde_control.RTDEControlInterface("10.213.55.123")
    
    rolling_average_count = 5
    rolling_average_joints = [[0, 0, 0, 0, 0, 0] for _ in range(rolling_average_count)]
    rolling_average_joints = np.array(rolling_average_joints)
    rolling_average_joints = rolling_average_joints.astype(np.float64)
    rolling_average_index = 0
    
    rolling_average_ready = False
    
    while (not stop_flag):  
        # Update the pose location if available
        base = 0
        if not pose_queue.empty():
            landmarks_data = pose_queue.get()
            #shoulder, elbow, wrist = pose_to_joint_angles(landmarks_data)
            shoulder, elbow, wrist = landmarks_data[0], landmarks_data[1], landmarks_data[2]
            #Joint positions for the UR10 robot
            # 0,0,0,0 is the home position which leaves the arm pointing straight up
            # 0 is the base and rotates around the up axis
            # 1 is the first joint and corresponds to the shoulder and rotates around the x or y axis
            # 2 is the second joint and corresponds to the elbow and rotates around the x or y axis
            # 3 is the third joint and corresponds to the wrist and rotates around the x or y axis

            vertical_vector = np.array([0, -1, 0])  
            upper_arm_vector = convert_to_left_handed(vector_between_points(shoulder, elbow))
            forearm_vector = convert_to_left_handed(vector_between_points(elbow, wrist))
            roll_reference_vector = np.array([0, 0, 1])
            yaw_reference_vector = np.array([1, 0, 0])

            # Subract 90 degrees from shoulder to match the zero position of the robot
            # Does not seem to be a way to get the zero position of the robot, so this is a workaround
            # Could potentially say that robot has to start in zero position and read that in the beginning
            shoulder_pitch_angle_radians = angle_between_vectors(upper_arm_vector, vertical_vector) - math.pi/2
            elbow_angle_radians = angle_between_vectors(upper_arm_vector, forearm_vector)
            shoulder_roll_angle_radians = angle_between_vectors(upper_arm_vector, roll_reference_vector)
            shoulder_yaw_angle_radians = angle_between_vectors(upper_arm_vector, roll_reference_vector)

            joint_positions = rtde_r.getActualQ()
            joint_positions[0] = shoulder_yaw_angle_radians
            joint_positions[1] = shoulder_pitch_angle_radians
            joint_positions[2] = -elbow_angle_radians
            
            rolling_average_joints[rolling_average_index] = joint_positions
            rolling_average_index = (rolling_average_index + 1) % rolling_average_count
            if rolling_average_index == 0:
                rolling_average_ready = True
                
            # print(rolling_average_joints)
            
            # if rolling_average_ready:
            #     average_joint_positions = np.mean(rolling_average_joints, axis=0)
                
            #     print(joint_positions)
            #     print(average_joint_positions)
            #     print(f'shoulder {math.degrees(shoulder_pitch_angle_radians)}, elbow {math.degrees(elbow_angle_radians)}')
                
            #     # period = rtde_c.initPeriod()
            #     rtde_c.moveJ(average_joint_positions, 1.05/4, 1.4/2, True)
            #     time.sleep(0.3)
            #     # rtde_c.waitPeriod(period)
                
            
            # print("HSHSHSHSH")    
            # rtde_c.moveJ_IK(np.asarray([0, 1, 0]), 1.05/4, 1.4/4)

        
        time.sleep(0.01)  # Sleep to avoid excessive CPU usage


def check_for_esc():
    global stop_flag
    keyboard.wait('esc')  # Wait until the Esc key is pressed
    print("Esc pressed, stopping...")
    stop_flag = True       # Set the stop flag

if __name__ == '__main__':
    pose_extraction_thread = threading.Thread(target=pose_extraction, daemon=True)
    pose_extraction_thread.start()

    simulation_thread = threading.Thread(target=robot_control, daemon=True)
    simulation_thread.start()

    check_for_esc()

    pose_extraction_thread.join()
    simulation_thread.join()

    print("Program terminated successfully")