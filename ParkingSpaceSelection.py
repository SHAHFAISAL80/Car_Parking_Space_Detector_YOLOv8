import cv2
import pickle
import numpy as np

try:
    with open(r'C:\Users\x\3D Objects\car_parking_space_detector_YOLOv8-master\car_parking_space_detector_YOLOv8-master\Space_ROIs', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

polygon_points = []  # List to store points of the polygon

def mouseClick(event, x, y, flags, params):
    global polygon_points, posList

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))

        if len(polygon_points) == 4:
            posList.append(polygon_points.copy())
            with open('carParkPos', 'wb') as f:
                pickle.dump(posList, f)
            polygon_points = []

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, polygon in enumerate(posList):
            if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0:
                posList.pop(i)
                with open('carParkPos', 'wb') as f:
                    pickle.dump(posList, f)
                break

while True:
    img = cv2.imread(r"C:\Users\x\3D Objects\car_parking_space_detector_YOLOv8-master\car_parking_space_detector_YOLOv8-master\ROI_Reference.png")
    for polygon in posList:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)

    for point in polygon_points:
        cv2.circle(img, point, 5, (0, 255, 0), -1)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
