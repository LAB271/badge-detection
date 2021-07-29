import cv2
import os
from datetime import datetime
from time import sleep

PATH_TO_RTSP_HIKVSION_DOME_CAMERA = 'rtsp://kplr:5hVUlm3S7o92@10.10.12.241/Streaming/channels/101'

cap = cv2.VideoCapture(PATH_TO_RTSP_HIKVSION_DOME_CAMERA)
now = datetime.now()
current_time = now.strftime(r"%d-%m-%Y_%H-%M-%S")
_, image = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join('output', 'Camera hikvision - {}.mp4'.format(current_time)),
                            fourcc, 10, (image.shape[1], image.shape[0]))

sleep(3)
while True:

    _, image = cap.read()
    out.write(image)

    if 0xFF == ord('q'):
        cap.release()
        out.release()
        break