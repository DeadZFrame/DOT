import cv2
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("Assets/Los Angeles 4K - California Glow - Scenic Drive - Trim.mp4")


class Object:
    object_ID = 0
    position = (0, 0)

    def __init__(self, object_ID, position):
        self.object_ID = object_ID
        self.position = position


# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []
    temp = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)

        temp.append((cx, cy))
        # print("FRAME N°", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    index = 0

    for cid in class_ids:
        obj = Object(cid, temp[index])
        center_points_cur_frame.append(obj)
        index += 1

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2.position[0] - pt.position[0], pt2.position[1] - pt.position[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2.position[0] - pt.position[0], pt2.position[1] - pt.position[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True

                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        if pt.object_ID == 2:
            cv2.circle(frame, pt.position, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt.position[0], pt.position[1] - 7), 0, 1, (0, 0, 255), 2)
        if pt.object_ID == 0:
            cv2.putText(frame, "Person", (pt.position[0], pt.position[1]), 0, 1, (0, 0, 255), 2)
        if pt.object_ID == 9:
            cv2.putText(frame, "Light", (pt.position[0], pt.position[1]), 0, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
