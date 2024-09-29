import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utilis import YOLO_Detection, drawPolygons, label_detection
import pickle


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load YOLO model and move it to the appropriate device
model = YOLO(r"C:\Users\x\3D Objects\car_parking_space_detector_YOLOv8-master\car_parking_space_detector_YOLOv8-master\yolov8n.pt")
model.to(device)

# Load the positions from the pickle file
with open(r'C:\Users\x\3D Objects\car_parking_space_detector_YOLOv8-master\car_parking_space_detector_YOLOv8-master\Space_ROIs', 'rb') as f:
    posList = pickle.load(f)

# Capture from camera or video
cap = cv2.VideoCapture(r"C:\Users\x\3D Objects\car_parking_space_detector_YOLOv8-master\car_parking_space_detector_YOLOv8-master\input_video\parking_space.mp4")  # Change to the appropriate source if not using a webcam
# get vcap property
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

                                        #### Main Loop ####
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO detection
        boxes, classes, names = YOLO_Detection(model, frame)

        # Collect points to determine if any detection is inside polygons
        detection_points = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = ((x1 + x2) / 2)
            center_y = ((y1 + y2) / 2)
            detection_points.append((int(center_x), int(center_y)))

        # Draw polygons with updated color based on detection status and make them transparent
        frame, occupied_count = drawPolygons(frame, posList, detection_points=detection_points)
        # Calculate available polygons
        available_count = len(posList) - occupied_count
        # Display the counts on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (int((width/2) - 200), 5), (int((width/2) - 40), 40), (250, 250, 250), -1)  # Rectangle dimensions and color
        # Put the current time on top of the black rectangle
        cv2.putText(frame, f"Fill Slots: {occupied_count}", (int((width/2) - 190), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95,
                    (50, 50, 50), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (int(width/2), 5), (int((width/2) + 175), 40), (250, 250, 250), -1)  # Rectangle dimensions and color
        # Put the current time on top of the black rectangle
        cv2.putText(frame, f"Free Slots: {available_count}", (int((width/2) + 10), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95,
                    (50, 50, 50), 1, cv2.LINE_AA)

        # Iterate through the results
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            detected_class = cls
            name = names[int(cls)]

            # Calculate the center point of the bounding box
            center_x = ((x1 + x2) / 2)
            center_y = ((y1 + y2) / 2)
            center_point = (int(center_x), int(center_y))

            # Define the color of the circle (BGR format)
            circle_color = (0, 120, 0)  # Green color in BGR
            cv2.circle(frame, center_point, 1, (255, 255, 255), thickness=2)

            # Determine the color of the bounding box based on detection location
            detection_in_polygon = False
            for pos in posList:
                matching_result = cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False)
                if matching_result >= 0:
                    detection_in_polygon = True
                    break

            if detection_in_polygon:
                label_detection(frame=frame, text=str(name), tbox_color=(50, 50, 50), left=x1, top=y1, bottom=x2, right=y2)
            else:
                label_detection(frame=frame, text=str(name), tbox_color=(100, 25, 50), left=x1, top=y1, bottom=x2, right=y2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

except:
    raise NotImplementedError