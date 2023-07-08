###
# ╭╴publisher.py
# ╰--> publish yolov8 inference data to websocket tunnel
# ╭╴Oportunitas (Taib Izzat Samawi); 07/Jul/2023
# ╰-@Barunastra_ITS
###

import cv2
import base64
import asyncio
import websockets
from websockets.server import serve
import sys
sys.path.append('./yolov8-object-recognition')
from yolov8_mylib import captureYOLOv8Inference

### define essential constants and variables BELOW this line
capture_loc = "/dev/video0" #video stream location
server_ip = "localhost" #server computer IP address
port_id = 64002 # port/lane of connection
### define essential constants and variables ABOVE this line

capture = cv2.VideoCapture(capture_loc)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
## capture footage from webcam, with .mjpg format

async def publishYOLOv8Inference(websocket, path):
    while True:
        confirm = await websocket.recv()
        print(f"command: {confirm}")
        ## wait for another request

        ret, frame = capture.read()
        if not ret:
            break
        ## capture current frame

        inf_frame = captureYOLOv8Inference(frame) # use YOLOv8 inference

        #cv2.imshow("after inference", frame)
        #if cv2.waitKey(1) == ord("q"):
        #    break
        # ## print inference data

        _, buffer = cv2.imencode('.jpg', inf_frame)
        frame_base64 = base64.b64encode(buffer)
        await websocket.send(frame_base64)
        print("Echoed back: alright heres another frame")
    ## run continuously (exit with ^C)
## asynchronously publish inference data

start_server = websockets.serve(publishYOLOv8Inference, "localhost", 64002)

print("server started")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
cv2.destroyAllWindows()