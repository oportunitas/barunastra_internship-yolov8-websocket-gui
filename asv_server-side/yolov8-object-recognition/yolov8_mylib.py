###
# yolov8-lib.py
#   base functions for yolov8@deepsparse functionalities
# Oportunitas (Taib Izzat Samawi); @07/Jul/2023
###

import cv2
import datetime
import base64
from ultralytics import YOLO
from deepsparse import Pipeline

#define constants and variables BELOW this line#
CONFIDENCE_TRESHOLD = 0.35

yolo_pipeline = Pipeline.create(
    task="yolov8",
    model_path="yolov8-object-recognition/weights/yolov8n-balls.onnx",
    num_cores=4
)
#define constants and variables ABOVE this line#

def captureYOLOv8Inference(frame):
    start = datetime.datetime.now()
    inference = yolo_pipeline(images=frame, iou_thres=0.5, conf_thres=0.2)
    for j in range (len(inference.boxes)):
        idx = 0
        for k in range (len(inference.boxes[j])):
            xmin = int(inference.boxes[j][k][0])
            ymin = int(inference.boxes[j][k][1])
            xmax = int(inference.boxes[j][k][2])
            ymax = int(inference.boxes[j][k][3])
            class_id = float(inference.labels[j][idx])
            idx += 1

            if (xmax-xmin <= 5 or ymax-ymin <= 5):
                continue

            COLOR = (0, 0, 0)
            class_name = "none"
            if (class_id == 1):
                class_name = "red_ball"
                COLOR = (0, 0, 255)
            elif (class_id == 0):
                COLOR = (0, 255, 0)
                class_name = "green_ball"

            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), COLOR, 1)
            cv2.putText(frame, class_name, (xmin-10,ymin-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)
    
    end = datetime.datetime.now()
    total = (end - start).total_seconds()

    fps = f"FPS: {1 / total:.2f}"
    FPS = 1 / total
    cv2.putText(frame, fps, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
    
    return frame