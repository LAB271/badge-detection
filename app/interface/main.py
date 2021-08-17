from threading import Thread
from time import sleep

from flask import Flask, render_template, Response

from app.detector.camera import SurveillanceCamera
from app.detector.models import PersonDetector, BadgeDetector, BadgeClassifier
from app.gstreamer import pipeline

app = Flask(__name__)

PATH_TO_RTSP_HIKVSION_DOME_CAMERA = 'rtsp://readonly:5hVUlm3S7o92@10.32.8.50/Streaming/channels/101'

# Increasing any of these values results in a better accuracy, however slower speeds
BUFFER = 7  # max image buffer capacity
OBJECT_LIFETIME = 10  # How long should the tracker still try to find a lost tracked person (measured in frames)
MAX_BADGE_CHECK_COUNT = 3  # How many times a full BUFFER should be checked before a person is declared to be an imposter

for _ in range(10):
    print("")

person_detection_model = PersonDetector().model
badge_detection_model = BadgeDetector().model
badge_classification_model = BadgeClassifier().model

hikvision = SurveillanceCamera('labs-hikvision', person_detection_model, badge_detection_model,
                               badge_classification_model, [1, 2, 3, 4, 5], PATH_TO_RTSP_HIKVSION_DOME_CAMERA, BUFFER,
                               OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT)
# hikvision_prerec = SurveillanceCamera('labs-preREC', person_detection_model, badge_detection_model, badge_classification_model, [1, 2, 3, 4, 5], os.path.join('output', 'Camera hikvision - 29-07-2021_20-13-07.mp4'), 10, 10, 4, 10, 3, interface=False)
# hikvision_prerec_copy = SurveillanceCamera('labs-preREC-----2', person_detection_model, badge_detection_model, badge_classification_model, [1, 2, 3, 4, 5], os.path.join('output', 'recordings', 'Camera labs-hikvision - 27-07-2021_19-14-41.mp4'), 10, 10, 4, 10, 3, interface=False)

print("-------------------------")

camera_list = [hikvision]


# scheduler_isRunning = False

def start_cameras():
    global camera_list
    for camera in camera_list:
        print("Attempting to start camera {}".format(camera.id))
        camera.start()
    print("Camera stream service started")
    pipeline.LOOP.run()


def update_cameras():
    while True:
        global camera_list
        for idx in range(len(camera_list)):
            camera = camera_list[idx]
            res = camera.update()
            if res is None:
                for attempt in range(10):
                    sleep(1)
                    res = camera.update()
                    if res is not None:
                        if attempt > 2:
                            print("Camera {} - Restarted".format(camera.id))
                        return
                camera_list.pop(idx)
            # print("Currently have {} cameras online".format(len(camera_list)))


@app.route('/')
def index():
    return render_template('carousel.html', list_len=len(camera_list), camera_list=camera_list)


@app.route('/video_feed/<cam_id>')
def video_feed(cam_id=None):
    if cam_id is not None:
        return Response(camera_list[int(cam_id)].get_frame_bytes(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    initializer = Thread(target=start_cameras)
    updater = Thread(target=update_cameras)

    initializer.start()
    sleep(2)
    updater.start()

    app.run(debug=False, use_reloader=False)
