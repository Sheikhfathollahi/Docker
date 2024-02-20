import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from model_detection.models.mtcnn import MTCNN

mtcnn = MTCNN()
modelFC = load_model(r"FCEFB0-0.h5")

# Iterate through the videos in the folder
folder_path = "./input"
video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
video_names = []
percentage_fake_list = []

for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    cap = cv2.VideoCapture(video_path)
    print(os.path.basename(video_path))
    video_names.append(os.path.basename(video_path))

    frame_count = 0
    Score = []
    while cap.isOpened():
        grabbed = cap.grab()
        if not grabbed:
            break

        ret, frame = cap.retrieve()
        if not ret:
            break

        if frame_count % 30 == 0:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for MTCNN
                boxes, probs = mtcnn.detect(frame)  # Detect faces using MTCNN
                if boxes is not None:
                    for box in boxes:
                        bbox = box.astype(int)
                        face_ROI = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_RGB2BGR)  # Convert back to BGR for model input

                        image = cv2.resize(face_ROI, (224, 224))
                        image = np.array(image)
                        image = np.expand_dims(image, axis=0)
                        predictions = modelFC.predict(image)
                        Score.append(predictions[0][0])

                else:
                    # MTCNN did not detect any faces, assign a score of 0.5
                    Score.append(0.5)

            except Exception as e:
                print(f"Error processing frame: {e}")

        frame_count += 1

    average_score = np.mean(Score) if Score else 0
    percentage_fake = 1 - average_score
    percentage_fake_list.append(percentage_fake)

video_names_and_feature = np.column_stack(
    (np.array(video_names).reshape(-1, 1), np.array(percentage_fake_list).reshape(-1, 1)))
Final_csv = pd.DataFrame(video_names_and_feature, columns=["Video Name", "Percentage Fake"])
Final_csv.to_csv('./output/submission.csv', sep=',', index=False, header=False)



