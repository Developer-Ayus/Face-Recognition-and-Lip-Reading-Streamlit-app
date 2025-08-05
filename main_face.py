import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# Load YOLOv11 pre-trained model
model = YOLO("yolo11n.pt")  # "yolov11s.pt" for better accuracy

# Path to the suspect's image
suspect_img_path = "suspect.jpg"

# Initialize video feed (0 for webcam, replace with drone feed URL if needed)
cap = cv2.VideoCapture(0)

# Function to check if a detected face matches the suspect
def is_suspect(face_crop):
    try:
        # Compare face with suspect image
        result = DeepFace.verify(face_crop, suspect_img_path, model_name="Facenet", enforce_detection=False)
        return result["verified"]  # True if suspect is detected
    except:
        return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 on the frame
    results = model(frame, verbose=False)  # Add verbose=False to suppress YOLO output   

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID (0 = person in COCO dataset)

            if cls == 0 and conf > 0.5:  # Only process detected people
                # Draw bounding box around detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract face region for verification
                face_crop = frame[y1:y2, x1:x2]

                # Check if detected person is the suspect
                if face_crop.size > 0 and is_suspect(face_crop):
                    cv2.putText(frame, "SUSPECT DETECTED!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for suspect

    # Display output
    cv2.imshow("YOLOv11 Suspect Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

