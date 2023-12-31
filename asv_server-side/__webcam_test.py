###
# ╭╴__webcam_test.py
# ╰--> dummy code | test webcam
# ╭╴Oportunitas (Taib Izzat Samawi); 07/Jul/2023
# ╰-@Barunastra_ITS
###

import cv2 # import openCV 2 python3 library

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
## capture webcam footage

while True:
    print("reading frame")
    ret, frame = capture.read()
    if not ret:
        print("failed to read frame")
    else:
        print("frame captured")
    new_frame = cv2.resize(frame, (640, 480))
    #read current frame
    print("showing frame")
    cv2.imshow('Webcam', new_frame)
    #show current frame
    if cv2.waitKey(1) == 27:
        break
    ##exits if <esq> button is pushed
##read currrent frame, continuously

capture.release()
cv2.destroyAllWindows()