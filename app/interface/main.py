from threading import Thread
from time import sleep
import psutil
import random

from flask import Flask, render_template, Response, request, redirect

from app.detector.camera import SurveillanceCamera
from app.detector.models import PersonDetector, BadgeDetector, BadgeClassifier
from app.gstreamer import pipeline

app = Flask(__name__)

PATH_TO_RTSP_HIKVSION_DOME_CAMERA = 'rtsp://readonly:5hVUlm3S7o92@10.32.8.50/Streaming/channels/101'
PATH_TO_RTSP_HIKVSION_360_CAMERA = 'rtsp://readonly:5hVUlm3S7o92@10.32.8.51/Streaming/channels/101'
# Increasing any of these values results in a better accuracy, however slower speeds
BUFFER = 5  # max image buffer capacity
OBJECT_LIFETIME = 5  # How long should the tracker still try to find a lost tracked person (measured in frames)
MAX_BADGE_CHECK_COUNT = 3  # How many times a full BUFFER should be checked before a person is declared to be an imposter

for _ in range(10):
    print("")

person_detection_model = PersonDetector().model
badge_detection_model = BadgeDetector().model
badge_classification_model = BadgeClassifier().model
print("-------------------------")

hikvision = SurveillanceCamera('labs-hikvision', person_detection_model, badge_detection_model,
                               badge_classification_model, [1, 2, 3, 4, 5], PATH_TO_RTSP_HIKVSION_DOME_CAMERA, BUFFER,
                               OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT)

camera_list = [hikvision]


# Device monitoring tools:
def monitor_device_info():
    while True:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()[2]
        yield "data: " + str(cpu) + str(ram) + "\n\n"
        sleep(2)

@app.route('/system_info')
def system_info():
    return Response(monitor_device_info(), mimetype='text/event-stream')

def monitor_camera_state():
    while True:
        for camera in camera_list:
            alert_code = camera.state['alert']
            if alert_code is not None:
                print("Should see an alert: code {}".format(alert_code))
                camera.updateState()
                yield "data: " + str(camera.id) + str("&&&") + str(alert_code) + "\n\n"
        sleep(15)
            

@app.route('/state_info')
def state_info():
    return Response(monitor_camera_state(), mimetype='text/event-stream')


def start_cameras():
    for camera in camera_list:
        print("Attempting to start camera {}".format(camera.id))
        camera.start()
    print("Camera stream service started")
    pipeline.LOOP.run()


def update_cameras():
    while True:
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
                #print("camera {} removed".format(camera.id))
                #camera_list.pop(idx)               # TODO: DELETE HASHTAGS AFTER TESTING
            yield "data: " + str(camera.id) + str("&&&") + str(res) + "\n\n"
        print("Currently have {} cameras online".format(len(camera_list)))
        
@app.route('/camera_manager')
def camera_manager():
    return Response(update_cameras(), mimetype='text/event-stream')



@app.route('/add_camera', methods=[ 'GET','POST'])
def add_camera():
    id = request.form.get("addCameraForm_id")
    path_to_stream = request.form.get("addCameraForm_url")
    allowed_badges = []
    for i in range(1, 6):
        badge = request.form.get("badge_"+str(i)) 
        if badge is not None:
            allowed_badges.append(i)

    camera = SurveillanceCamera(id, person_detection_model, badge_detection_model,
                               badge_classification_model, allowed_badges, path_to_stream, 
                               BUFFER, OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT)
    camera.start()
    camera_list.append(camera)

    #return ('', 204)
    return index()

@app.route('/remove_camera', methods=['GET', 'POST'])
def remove_camera():
    id = request.form.get("removeCameraForm_id")
    for idx in range(len(camera_list)):
        if camera_list[idx].id == id:
            camera_list.pop(idx)
            break

    return index()

# TODO: setup the sliders in html
'''@app.route('/update_settings', methods=['GET', 'POST'])
def update_settings():
    buffer_size = request.form.get("buffer_slider")
    object_lifetime = request.form.get("lifetime_slider")
    max_badge_check_count = request.form.get("maxcheck_slider")

    for camera in camera_list:
        camera.buffer_size = buffer_size
        camera.object_lifetime = object_lifetime
        camera.max_badge_check_count = max_badge_check_count
    
    return ('', 204)'''


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('carousel.html', list_len=len(camera_list), camera_list=camera_list)


@app.route('/video_feed/<cam_id>')
def video_feed(cam_id=None):
    if cam_id is not None:
        return Response(camera_list[int(cam_id)].get_frame_bytes(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    initializer = Thread(target=start_cameras)
    initializer.start()
    sleep(2)

    app.run(debug=True, use_reloader=True, host="127.0.0.1", port="5000")
