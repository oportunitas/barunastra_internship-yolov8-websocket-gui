###
# ╭╴displayer.py
# ╰--> display captured yolov8 inference data from websocket tunnel
# ╭╴Oportunitas (Taib Izzat Samawi); 07/Jul/2023
# ╰-@Barunastra_ITS
###

import cv2
import base64
import asyncio
import websockets
from websockets.sync.client import connect
import numpy
import datetime
## import all dependencies

### define essential constants and variables BELOW this line
server_ip = "localhost" #server computer IP address
port_id = 64002 # port/lane of connection
### define essential constants and variables ABOVE this line

adr = ("ws://" + server_ip + ":" + str(port_id) + "/") # full address of ws tunnel

async def displayYOLOv8Inference():
    try:
        async with websockets.connect(adr, ping_interval=0, timeout=0) as websocket:
            first_comm = await websocket.send("first frame request") # tell server to request a frame
            while True:
                start = datetime.datetime.now() # store start datetime

                frame_base64 = await websocket.recv() 
                ## wait and receive binary stream from asv

                frame_bytes = base64.b64decode(frame_base64)
                frame_array = numpy.frombuffer(frame_bytes, dtype=numpy.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                ## decode binary stream from websocket tunnel into image

                cv2.imshow('Live Stream', frame)
                if cv2.waitKey(1) == 27: # esq button to quit
                    break
                ## show image

                #confirm = await websocket.send("frame request") # tell server to request another frame

                end = datetime.datetime.now()
                total = (end - start).total_seconds()
                #print(f"FPS: {1 / total:.2f}", end="\r")

            ## run continuously (exit with ^C)
        ## __using specified address
    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed. Reconnecting...")
        await asyncio.sleep(0)
        ## notify reconnection then reconnect
    ## try to establish a connection and display frame, except if disconnected, connect again
## asynchronously display inference data

while True:
    asyncio.get_event_loop().run_until_complete(displayYOLOv8Inference()) # run operation
## dummy while loop, prevents unintended close loop

cv2.destroyAllWindows() # close all cv2 windows