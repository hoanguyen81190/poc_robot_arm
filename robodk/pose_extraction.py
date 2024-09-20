import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
import threading, queue
import random
import json

# Initialize MediaPipe Pose

def on_connect(client, userdata, flags, rc, properties):
    print(f"Connected with result code {rc}-----------------------------")
    client.subscribe('testmqtt')
    #client.subscribe(MQTT_METADATA['motiontopic'])

def on_message(client, userdata, message):
    print("I got a message")

client_id = f'python-mqtt-{random.randint(0, 1000)}'
#client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport='websockets')
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="tcp")

client.on_connect = on_connect
client.on_message = on_message

# Was port 9001
client.connect('localhost', 9001, 60)  # 60 is waiting interval
client.loop_start()

# Define the MQTT topic
mediapipe_topic = 'mediapipe'

def pose_extraction():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Open the webcam
    # Was capture id 1
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
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
            landmarks_data = [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]
            # Publish landmarks data to MQTT
            client.publish(mediapipe_topic, json.dumps(landmarks_data))
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


if __name__ == '__main__':
    pose_extraction_thread = threading.Thread(target=pose_extraction, )
    pose_extraction_thread.start() 