from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from scipy.spatial.transform import Rotation as R
import time

import cv2
import mediapipe as mp
import threading, queue
import keyboard
import math
import numpy as np

from Utils.utils import clamp_angle, convert_mediapipe_to_coppeliasim, convert_to_left_handed, deadband_control, getJointAnglesFromPose, map_to_range, normalize_angle, testForwardIKPY
from Utils import PIDController, ExponentialSmoother

pose_queue = queue.Queue(1)

# Extract the arm joints by default
def pose_extraction(desired_landmark_ids=[12, 14, 16]):
    print("Func Inputs: ", desired_landmark_ids)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Open the webcam
    # Was capture id 1
    cap = cv2.VideoCapture(0)

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

def cartesian_to_cylindrical_3d(data):
    x, y, z = data[0], data[1], data[2]
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta, z

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

    #sim.setStepping(True)

    joint_names = [
        'UR10_joint1',
        'UR10_joint2',
        'UR10_joint3',
        'UR10_joint4',
        'UR10_joint5',
        'UR10_joint6'
    ]

    #sim.startSimulation()

    joint_handles = []
    for name in joint_names:
        handle = sim.getObjectHandle(name)
        if handle != -1:
            joint_handles.append(handle)
            #Configure speed of joint
            #defaultVelocity = sim.getJointTargetVelocity(handle)
            #print(f'Default velocity for {handle}: {defaultVelocity}')
            #sim.setJointTargetVelocity(handle, 10)
            velocity = 2.0 #-- Set desired velocity (in rad/s for revolute joints)
            acceleration = 5.0 #-- Set desired acceleration

            sim.setJointTargetVelocity(handle, velocity)
            sim.setObjectFloatParameter(handle, sim.jointfloatparam_maxvel, velocity)
            sim.setObjectFloatParameter(handle, sim.jointfloatparam_maxaccel, acceleration)
            print(f'Handle for {name} retrieved successfully')
        else:
            print(f'Failed to get handle for {name}')

    
    # PID controller for end-effector control
    pid_controller = PIDController.PIDController(Kp=0.15, Ki=0.01, Kd=0.001)

    # Initialize exponential smoother for the target position
    alpha = 0.2
    smoother = ExponentialSmoother.ExponentialSmoother(alpha)
    smoother_joint = ExponentialSmoother.ExponentialSmoother(alpha)

    prev_time = time.time()

    bUsePID = False
    bUseDeadbandThresholding = True
    prevDeadbandPos = np.asarray([0,0,0])

    # Continuously step the coppelia simulation
    while (True):
        if keyboard.is_pressed('esc'):  # Check if ESC key is pressed
            print("ESC key pressed. Stopping simulation.")
            break
        
        # Update the pose location if available 
        # TODO find out and document the coordinate systems of the mediapipe 
        #   and coppelia sim and the robot arm
        if not pose_queue.empty():
            dt = time.time()-prev_time

            landmarks_data = pose_queue.get_nowait()
            landmarks_target_position = np.mean(landmarks_data,axis=0)[0:3]

            # Swap axis so it makessense in terms of controlling the robot arm
            re_mapped = landmarks_target_position.copy()
            re_mapped[0] = landmarks_target_position[2]
            re_mapped[1] = landmarks_target_position[0]
            re_mapped[2] = landmarks_target_position[1]

            # Map the input ranges so they are more meaningful for cotrolling the robot arm
            re_mapped_2 = re_mapped.copy()
            re_mapped_2[1] = map_to_range(re_mapped[1],0,1,-1.3,1.3)
            re_mapped_2[2] = map_to_range(1-re_mapped[2],0,1,0,1.3)
            #print("Re-mapped to control robot arm: ", re_mapped_2)
            
            # Apply exponential smoothing 
            smoothed_pos = smoother.update(re_mapped_2)
            #print("Smoothed pos: ", smoothed_pos)


            # TMP for testing one axis at a time
            smoothed_pos = np.asarray([0,smoothed_pos[1],smoothed_pos[2]])

            max_reach = 1.3
            target_position_in = np.clip(smoothed_pos, -max_reach, max_reach)
            
            #joint_angles = getJointAnglesFromPose(target_position_in, plotFig=False)
            #joint_angles = [clamp_angle(normalize_angle(angle)) for angle in joint_angles]


            #target_orientation = np.eye(3) #TODO Update to use the angles between the joints later
            #target_quaternion = R.from_matrix(target_orientation).as_quat()

            # Get the current end-effector position and orientation from the simulation
            end_effector_pose = np.asarray(sim.getObjectPose(joint_handles[-1]))

            #target_pose = np.asarray([*target_position,*target_quaternion])

            #control_signal = pid_controller.update(target_pose, end_effector_pose, dt)

            if bUsePID:
                control_signal = pid_controller.update(target_position_in, end_effector_pose[0:3], dt)
                new_target_position = control_signal[:3] #target_position_in + control_signal[:3]
            else:
                new_target_position = target_position_in 
            #new_target_orientation = target_orientation + control_signal[3:]

            #print("Input target pos: ", target_position_in)
            #print("End effector pos: ", end_effector_pose[0:3])

            if bUseDeadbandThresholding:
                # TODO update this when starting to take depth into account
                #end_effector_pose[0] = 0
                new_target_position, applied = deadband_control(new_target_position, prevDeadbandPos, deadband_threshold=0.05)
                if applied:
                    prevDeadbandPos = new_target_position

            #Slice to remove the fixed joints immidiately to avoid consufion down the line
            joint_angles = getJointAnglesFromPose(new_target_position)[1:-1]#target_position)
            joint_angles = np.asarray([clamp_angle(normalize_angle(angle)) for angle in joint_angles])

            joint_angles = smoother_joint.update(joint_angles)

            # Rate limit the angles to increase stability
            #max_angle_change_per_step = 0.1  # Limit to 0.1 radians per time step
            #joint_angles = np.clip(
            #    joint_angles, 
            #    joint_angles - max_angle_change_per_step, 
            #    joint_angles + max_angle_change_per_step
            #)

            #shoulder, elbow, wrist = pose_to_joint_angles(landmarks_data)
            #Joint positions for the UR10 robot
            # 0,0,0,0 is the home position which leaves the arm pointing straight up
            # 0 is the base and rotates around the up axis
            # 1 is the first joint and corresponds to the shoulder and rotates around the x or y axis
            # 2 is the second joint and corresponds to the elbow and rotates around the x or y axis
            # 3 is the third joint and corresponds to the wrist and rotates around the x or y axis
            #joint_positions = [base, shoulder, elbow, wrist, 0, 0]  # Example positions in radians

            for i, handle in enumerate(joint_handles):
                sim.setJointTargetPosition(handle, joint_angles[i]) #NB do not forget to skip the base joint and the end effector joint from the IK chain. They are fixed and not used when controlling the robot. 

            prev_time = time.time()

        # Step the simulation
        #simulation_time = 0
        #while simulation_time < 5:  # Run for 5 seconds
        #print(f'Simulation time: {simulation_time:.2f} [s]')
        #sim.step()
        #simulation_time = sim.getSimulationTime()
        time.sleep(0.01)  # Sleep to avoid excessive CPU usage

    # Stop the simulation
    #sim.stopSimulation()

if __name__ == '__main__':
    landmark_ids = [16,18,20]
    pose_extraction_thread = threading.Thread(target=pose_extraction, args=(landmark_ids,))
    pose_extraction_thread.start()

    simulation_thread = threading.Thread(target=simulation, )
    simulation_thread.start()