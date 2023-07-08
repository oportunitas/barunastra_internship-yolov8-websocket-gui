###
# ╭╴yolov8-lib.py
# ╰--> base functions for yolov8@deepsparse functionalities
# ╭╴Oportunitas (Taib Izzat Samawi); 07/Jul/2023
# ╰-@Barunastra_ITS
###

import cv2
import datetime
import base64
from ultralytics import YOLO
from deepsparse import Pipeline
## import all dependencies

### define constants and variables BELOW this line#
my_conf_thres = 0.2
my_iou_thres = 0.6

yolo_pipeline = Pipeline.create(
    task="yolov8",
    model_path="yolov8-object-recognition/weights/yolov8n-balls.onnx",
    num_cores=4
)
### define constants and variables ABOVE this line#

def idToName(class_id):
    switcher = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }
    return switcher.get(int(class_id), "nothing")
## dummy function for testing against coco dataset

def captureYOLOv8Inference(frame):
    start = datetime.datetime.now() #store current date-time
    inference = yolo_pipeline(images=frame, iou_thres=my_iou_thres, conf_thres=my_conf_thres) #run YOLOv8 on the frame given

    for k in range (len(inference.boxes[0])):
        xmin = int(inference.boxes[0][k][0])
        ymin = int(inference.boxes[0][k][1])
        xmax = int(inference.boxes[0][k][2])
        ymax = int(inference.boxes[0][k][3])
        class_id = float(inference.labels[0][k])
        ## assign variables based on bounding box data

        if (xmax-xmin <= 5 or ymax-ymin <= 5):
            continue
        ## if ball is very small, ignore

        COLOR = (0, 0, 0)
        class_name = "none"
        ## set default values

        if (class_id == 1):
            class_name = "red_ball"
            COLOR = (0, 0, 255)
        elif (class_id == 0):
            COLOR = (0, 255, 0)
            class_name = "green_ball"
        ## assign box color

        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), COLOR, 1)
        cv2.putText(frame, class_name, (xmin-10,ymin-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)
        ## draw box and name on the image given
    ## iterate over every bounding box detected, stored at inference.boxes[0] for deepsparse
    
    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    fps = f"FPS: {1 / total:.2f}"
    ## calculate FPS for current calculation

    cv2.putText(frame, fps, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
    ## print FPS value
    
    return frame
## given an image, label that image with bounding boxes according to a YOLOv8 model