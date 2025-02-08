import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load the Haar cascade classifier for face detection
face_cap = cv2.CascadeClassifier("C:/Users/praro/OneDrive/Documents/DAA/haarcascade_frontalface_default.xml")

# Load pre-trained deep learning models for age and gender
age_proto = "deploy_age.prototxt"
age_model = "age_net (3).caffemodel"
gender_proto = "deploy_gender.prototxt"
gender_model = "gender_net (2).caffemodel"

# Load models
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Age and Gender categories
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Function to start webcam feed and perform detection
def start_webcam():
    # Start video capture
    video_cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not video_cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, video_data = video_cap.read()

        if not ret:
            print("Error: Failed to capture video frame.")
            break

        # Convert frame to grayscale
        col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cap.detectMultiScale(col, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the face region of interest (ROI)
            face_roi = video_data[y:y + h, x:x + w].copy()

            # Preprocess the face ROI for deep learning models
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Display age and gender on frame
            text = f"{gender}, {age}"
            cv2.putText(video_data, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the video frame
        cv2.imshow("Age & Gender Detection", video_data)

        # Press 'a' to exit
        if cv2.waitKey(10) & 0xFF == ord("a"):
            break

    # Release resources
    video_cap.release()
    cv2.destroyAllWindows()

# print("Press 's' to start the webcam.")
# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("s"):
#         start_webcam()  # Start the webcam if 's' is pressed
#         break
#     elif key == ord("a"):
#         break  # Exit if 'a' is pressed while waiting for 's'

# Tkinter GUI setup
root = tk.Tk()
root.title("Age & Gender Detection")
root.geometry("300x150")

# Function to start webcam when the button is pressed
def on_button_click():
    # Start the webcam feed
    start_webcam()

# Add a button in Tkinter to start webcam
start_button = tk.Button(root, text="Start Webcam", command=on_button_click, font=("Arial", 14))
start_button.pack(pady=50)

# Run the Tkinter event loop
root.mainloop()
