
from app.detector.camera import SurveillanceCamera
from app.detector.models import PersonDetector, BadgeDetector, BadgeClassifier
import os
from flask import Flask, render_template, Response
from threading import Thread
from time import sleep

#Initialize the Flask app
app = Flask(__name__)

PATH_TO_RTSP_HIKVSION_DOME_CAMERA = 'rtsp://kplr:5hVUlm3S7o92@10.10.12.241/Streaming/channels/101'
# Increasing any of these values results in a better accuracy, however slower speeds

BUFFER = 10  # max image buffer capacity
OBJECT_LIFETIME = 5  # How long should the tracker still try to find a lost tracked person (measured in frames)
MAX_BADGE_CHECK_COUNT = 3  # How many times a full BUFFER should be checked before a person is declared to be an imposter

for _ in range(10):
    print("")

person_detection_model = PersonDetector().model
badge_detection_model = BadgeDetector().model
badge_classification_model = BadgeClassifier().model

hikvision = SurveillanceCamera('labs-hikvision', person_detection_model, badge_detection_model, badge_classification_model, [2,3,4,5], PATH_TO_RTSP_HIKVSION_DOME_CAMERA, 10, 6, BUFFER, OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT, interface = False)
#hikvision_prerec = SurveillanceCamera('labs-preREC', person_detection_model, badge_detection_model, badge_classification_model, [1, 2, 3, 4, 5], os.path.join('output', 'Camera hikvision - 29-07-2021_20-13-07.mp4'), 10, 10, 4, 10, 3, interface=False)
#hikvision_prerec_copy = SurveillanceCamera('labs-preREC-----2', person_detection_model, badge_detection_model, badge_classification_model, [1, 2, 3, 4, 5], os.path.join('output', 'recordings', 'Camera labs-hikvision - 27-07-2021_19-14-41.mp4'), 10, 10, 4, 10, 3, interface=False)

print("-------------------------")

camera_list = [hikvision]  
scheduler_isRunning = False  


def read_frames():
    while True:
        global camera_list
        for idx in range(len(camera_list)):
            camera = camera_list[idx]
            camera.read_frame()

def update_cameras():
    while True:
        global camera_list
        print("Currently have {} cameras online".format(len(camera_list)))
        for idx in range(len(camera_list)):
            camera = camera_list[idx]
            #print("updating camera {}".format(camera.id))
            res = camera.update()
            if res is None:
                for attempt in range(10):
                    sleep(1)
                    print(attempt)
                    res = camera.update()
                    if res is not None:
                        print("Reset")
                        return
                camera_list.pop(idx)


@app.route('/')
def index():
    return render_template('index.html', list_len=len(camera_list), camera_list=camera_list)

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id=None):
    if cam_id is not None:
        return Response(camera_list[int(cam_id)].get_frame_bytes(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings')
def settings():
    return render_template('settings.html')


if __name__ == "__main__":
    for camera in camera_list:
        camera.start()
    frame_reader = Thread(target=read_frames)
    updater = Thread(target=update_cameras)
    frame_reader.start()
    updater.start()

    app.run(debug=True, use_reloader=True)
    

        