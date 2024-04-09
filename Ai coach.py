#Importing needed models

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#Determine the dataset path

# dataset = 'All data'
model_save_path = 'The saved model/model.keras'

#Load the model

model = tf.keras.models.load_model(model_save_path)

#model test by video

video_path = '29_Demo (2).mp4'
cap = cv2.VideoCapture(video_path)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame with MediaPipe Pose Detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    # Check if pose landmarks are detected
    if result.pose_landmarks:
        # Extract pose landmarks
        landmarks = [[lm.x, lm.y] for lm in result.pose_landmarks.landmark]

        # Preprocess landmarks (reshape, convert to numpy array, etc.)
        # Example:
        landmarks_array = np.array(landmarks, dtype=np.float32)
        landmarks_array = landmarks_array[np.newaxis, ...]  # Add batch dimension

        # Use the model to make predictions
        predictions = model.predict(landmarks_array)
        predicted_class = "nothing"
        # Example: Print the predicted class
        min_value = 0.5
        if np.max(predictions) >= min_value:
            predicted_class = ""
            if np.argmax(predictions) == 0:
                predicted_class = "crunches false"
                m = 0
            elif np.argmax(predictions) == 1:
                predicted_class = "crunches true"
                m = 1
            elif np.argmax(predictions) == 2:
                predicted_class = "push up false"
                m = 0
            elif np.argmax(predictions) == 3:
                predicted_class = "push up true"
                m = 1
            elif np.argmax(predictions) == 4:
                predicted_class = "squat false"
                m = 0
            elif np.argmax(predictions) == 5:
                predicted_class = "squat true"
                m = 1
        else:
            predicted_class = "nothing"   
        if m == 0:     
          cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
          cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  

    # Display the frame
    cv2.imshow("output",frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()



