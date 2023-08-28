###
# ╭╴publisher.py
# ╰--> publish yolov8 inference data to websocket tunnel
# ╭╴Oportunitas (Taib Izzat Samawi); 07/Jul/2023
# ╰-@Barunastra_ITS
###

print("starting publisher.py")

import cv2
import base64
import asyncio
import websockets
import datetime
import os
from websockets.server import serve
import sys
sys.path.append('./yolov8-object-recognition')
print("check")
from yolov8_mylib import captureYOLOv8Inference
print("got all dependencies")
## import all dependencies

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
        start = datetime.datetime.now() # store start datetime

        #confirm = await websocket.recv()
        ## wait for another request

        cap_start = datetime.datetime.now()
        ret, frame = capture.read()
        if not ret:
            break
        ## capture current frame
        cap_end = datetime.datetime.now()

        inf_start = datetime.datetime.now()
        inf_frame = captureYOLOv8Inference(frame) # use YOLOv8 inference
        inf_end = datetime.datetime.now()

        #cv2.imshow("after inference", frame)
        #if cv2.waitKey(1) == ord("q"):
        #    break
        # ## print inference data

        buf_start = datetime.datetime.now()
        _, buffer = cv2.imencode('.jpg', inf_frame)
        frame_base64 = base64.b64encode(buffer)
        await websocket.send(frame_base64)
        buf_end = datetime.datetime.now()
        
        end = datetime.datetime.now()

        cap_total = (cap_end - cap_start).total_seconds()
        buf_total = (buf_end - buf_start).total_seconds()
        inf_total = (inf_end - inf_start).total_seconds()
        total = (end - start).total_seconds()

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"capture FPS: {1 / cap_total:.2f}")
        print(f"inference FPS: {1 / inf_total:.2f}")
        print(f"buffer FPS: {1 / buf_total:.2f}")
        #print(f"process FPS: {1 / total:.2f}")

    ## run continuously (exit with ^C)
## asynchronously publish inference data

start_server = websockets.serve(publishYOLOv8Inference, "localhost", 64002)

print("server started")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
cv2.destroyAllWindows()