# image_demo.py: Testing demonstration.

import cv2
import os
import torch
import numpy as np
from models.cnn import CNN

# Paths for models and emotion labels
emotion_model_path = 'results/fer2013plus/fer2013plus_cnn/2nd/model_cnn_fer2013plus.pth'
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_labels = { # Mapping of emotion classes to human-readable labels
    0: "happy",
    1: "surprise",
    2: "sad",
    3: "angry",
    4: "disgust",
    5: "fear",
    6: "neutral"
}

# Load the pre-trained emotion classification model
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
emotion_classifier = CNN().to(device)
emotion_classifier.load_state_dict(torch.load(emotion_model_path, map_location=device, weights_only=False))
emotion_classifier.eval()

# Load Haar Cascade for face detection
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_target_size = (48, 48)  # Expected input size for the emotion classifier model

def preprocess_input(x, v2=True):
    # Preprocess the image input for the emotion classification model.
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def general_predict(imggray, imgcolor):
    # Detect faces, preprocess them, and predict emotions.
    faces = face_detection.detectMultiScale(imggray, 1.3, 5)
    res = []
    if len(faces) == 0:
        print('No face detected')
        return None
    else:
        for face_coordinates in faces:
            x1, y1, width, height = face_coordinates
            x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height
            gray_face = imggray[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, emotion_target_size) 
            except Exception as e:
                print(f"Error resizing face: {e}")
                continue

            # Preprocess face
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.repeat(gray_face, 3, axis=0)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face_tensor = torch.tensor(gray_face, dtype=torch.float32).to(device)

            # Predict emotion
            with torch.no_grad():
                emotion_prediction = emotion_classifier(gray_face_tensor)
                emotion_prediction = torch.nn.functional.softmax(emotion_prediction, dim=1)

            emotion_label_arg = torch.argmax(emotion_prediction, dim=1).item()
            confidence = emotion_prediction[0, emotion_label_arg].item()  # Confidence for the predicted emotion
            
            res.append([emotion_label_arg, x1, y1, x2, y2, confidence])

    return res

def save_predict(imgurl, targeturl='images_labeled/happy1.png'):
    # Detect faces and emotions, annotate the image, and save it.
    imggray = cv2.imread(imgurl, 0)
    imgcolor = cv2.imread(imgurl, 1)
    if imggray is None or imgcolor is None:
        print(f"Error reading image: {imgurl}")
        return

    ress = general_predict(imggray, imgcolor)
    if ress is None:
        print('No face detected, no image saved')
        return

    for res in ress:
        emotion_index = res[0]
        confidence = res[5]  # Extract confidence
        label = f"{emotion_labels[emotion_index]} ({confidence*100:.2f}%)"  # Add confidence percentage

        lx, ly, rx, ry = res[1], res[2], res[3], res[4]
        cv2.rectangle(imgcolor, (lx, ly), (rx, ry), (0, 255, 0), 2)
        cv2.putText(imgcolor, label, (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(targeturl), exist_ok=True)
    cv2.imwrite(targeturl, imgcolor)
    print(f"Predicted image saved to {targeturl}")

# Test the function with a sample image
save_predict('images_original/happy1.jpg')