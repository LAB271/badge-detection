from datetime import datetime
from statistics import mode

import cv2
import numpy as np
from gi.repository import GstApp
from torch import no_grad

from app.detector.person import Person
from app.detector.utils import normalise_bbox, image_loader, print_alert, flatten_list, \
    tensor_to_image, bytes_to_array
from app.gstreamer import pipeline
from sort.sort import Sort
from threading import Thread


class SurveillanceCamera(object):
    count = 0

    def __init__(self, id, face_predictor, badge_predictor, badge_classifier, allowed_badges, path_to_stream,
                 buffer_size,
                 object_lifetime, max_badge_check_count):

        self.id = id
        self.buffer_size = buffer_size
        self.object_lifetime = object_lifetime
        self.max_badge_check_count = max_badge_check_count + buffer_size
        self.cam_url = path_to_stream

        '''if self.record is not None:
            now = datetime.now()
            current_time = now.strftime(r"%d-%m-%Y_%H-%M-%S")
            _, image = self.cap.read()
            #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.out = cv2.VideoWriter()
            self.out.open(os.path.join(self.record, 'Camera {} - {}.avi'.format(self.id, current_time)),
                                       fourcc, wanted_fps, (image.shape[1], image.shape[0]))'''

        self.mot_tracker = Sort(max_age=object_lifetime)
        self.face_predictor = face_predictor
        self.badge_predictor = badge_predictor
        self.badge_classifier = badge_classifier
        self.allowed_badges = allowed_badges
        self.tracked_person_list = []
        self.last_read_frame = None
        self.last_updated_frame = None
        self.orig_image = None
        SurveillanceCamera.count += 1
        self.fps = FPS()
        self.read_frames_fps = FPS()
        self.frame_id = 0

    def update(self):

        # Start FPS count engine
        if self.frame_id == 0:
            self.fps.start()

        # Stop service if stream ended
        if self.last_read_frame is None:
            # Implement some sort of restart system here. Security cameras should not ever be turned off
            return None

        # Avoid processing the same frame again
        if self.last_read_frame['scanned']:
            return True
        
        self.last_read_frame['scanned'] = True

        init_start_time = datetime.now()
        # Convert PIL image to a np array
        #self.orig_image = np.array(self.last_read_frame['image'])

        self.orig_image = self.last_read_frame['image']
        print("Time taken to convert image: {}".format(datetime.now()-init_start_time))

        # Person detection and tracking
        start_time = datetime.now()
        detected_faces, _, face_scores = self.face_predictor.predict(self.orig_image, prob_threshold=0.9)  # (image, candidate_size/2, threshold)
        print("Time taken to detect {} persons: {}".format(len(self.tracked_person_list), datetime.now()-start_time))
        start_time = datetime.now()
        self.track_persons(detected_faces, face_scores)
        print("Time taken to track {} persons: {}".format(len(self.tracked_person_list), datetime.now()-start_time))
            
        
        start_time = datetime.now()
        for person in self.tracked_person_list:
            person.badge_check_count += 1
            # Self-destruction of Person objects (if they're not being used)
            if not person.isAlive():
                for index in range(len(self.tracked_person_list)):
                    if self.tracked_person_list[index] == person:
                        self.tracked_person_list.pop(index)
                        break
                continue

            # Look for a badge if the person hasn't been found to have one yet
            # TODO: if a badge was not found, after the decision has been made, continue looking for it. And if found, try to classify it. 
            if person.badge is None or person.badge_number is None:
                person_cutout = person.getImage(person.getBufferOppacity('person cutouts')-1, as_tensor=True)
                print(person.getBufferOppacity('person cutouts'))
                badge_found, scan_data = self.detect_badge(person_cutout, threshold=0.5)
                print("Score: {}".format(scan_data['score']))
                if badge_found:
                    person.addScanDataToBuffer(scan_data)
                    person.badge = True

                '''# If there's enough data in the buffer to proceed, make a conclusion about the badges
                if person.getBufferOppacity('scanned') > 1:

                    score_list = person.getBuffer('scanned scores')
                    detection_results = self.evaluate_detected_badges(score_list)

                    if detection_results is not None:
                        person.badge = True
                        # person.badge_score = detection_results

                    # person.badge_check_count+=1'''

            # if a person HAS a badge, but it's not yet known which one, initiate the classifier module
            if person.badge and person.badge_number is None:
                classified_badges = {'score': [], 'label': []}
                for buffer in range(person.getBufferOppacity('scanned') - 1):
                    badge_cutout = person.getImage(buffer, as_tensor=True, value='scan')
                    badge_classified, scan_data = self.classify_badge(badge_cutout, threshold=0.6)
                    if badge_classified:
                        # Add classified badge number and score to dictionary
                        classified_badges['score'].append(scan_data['score'])
                        classified_badges['label'].append(scan_data['label'])
                        print(classified_badges)
                person.badge_number, person.badge_score = self.evaluate_classified_badges(classified_badges["label"],
                                                                                          classified_badges["score"])
                # TODO: Add scanned data to buffer; each round scan again and evaluate together with data in buffer

                if person.badge_score is None:
                    print('Failed to classify badge for person {}'.format(person.id))
                    print("Badge Check: {}/{}".format(person.badge_check_count, self.max_badge_check_count))
                    print(classified_badges["label"],classified_badges["score"])
                    if person.badge_check_count == self.max_badge_check_count:
                        person.badge = False
                        person.badge_number = 0
                        print_alert(2, self.id, person.id)

                elif person.badge_score > 0.8:
                    # Steps to take when the system is confident in the result of the badge detection models
                    if person.badge_number in self.allowed_badges:
                        person.clearBuffer()
                    else:
                        # person.badge = False
                        print_alert(1, self.id, person.id)
                        # ALERT: this person is definitely not supposed to be here
                else:
                    # Steps to take when the system is NOT confident in the result of the badge detection models
                    # For now - check for the badge again
                    pass

            # if the badge has been checked enough times and not found, report that badge was not found.
            if person.badge is None and person.badge_check_count == self.max_badge_check_count:
                # person.badge_check_count += 1
                person.badge = False
                person.badge_number = 0
                print_alert(0, self.id, person.id)  # TODO: Detection confidence results

            
            # print("Badge checked for {} time".format(person.badge_check_count))
        print("Time taken to check {} persons: {}".format(len(self.tracked_person_list), datetime.now()-start_time))
        start_time = datetime.now()
        self.frame_id += 1
        self.fps.update()
        self.fps.stop()
        print("update_module fps: %s" % self.fps.fps())
        cv2.putText(self.orig_image, ('fps: {}'.format(round(self.fps.fps(), 2))), (100, self.orig_image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
        self.last_updated_frame = self.orig_image
        print("Time taken to complete final calculations: {}".format(datetime.now()-start_time))
        print("Time taken to complete full loop: {}".format(datetime.now()-init_start_time))
        return True

    def track_persons(self, detected_faces, face_scores):
        # Basic image prep
        # self.orig_image = cv2.flip(self.orig_image, 0)
        image_dimensions = self.orig_image.shape  # (h,w,c)

        # If any persons were detected, track them, and add cutout images to their BUFFER
        if len(detected_faces) != 0:
            # Formatting the arrays for (deep)SORT into a numpy array that contains lists of (x1,y1,x2,y2,score)
            face_data = []
            for i in range(len(detected_faces)):
                # Changing the coordinates to bound the body instead of the face but also ensuring it doesn't go outside of the image bounds
                # Head to body ratio is ~ 1:4 - 1:8. That can be used to mark the required body size knowing the head measurements
                ratioW = detected_faces[i][2] - detected_faces[i][0]
                ratioH = (detected_faces[i][3] - detected_faces[i][1]) * 6
                detected_faces[i][0] = int(detected_faces[i][0]) - ratioW
                detected_faces[i][1] = int(detected_faces[i][1])
                detected_faces[i][2] = int(detected_faces[i][2]) + ratioW
                detected_faces[i][3] = detected_faces[i][1] + ratioH
                detected_faces[i] = normalise_bbox(detected_faces[i], image_dimensions)
                temp = np.append(detected_faces[i], face_scores[i])
                face_data.append(temp)
            face_data = np.array(face_data)

            # Calling the person tracker
            track_bbs_ids = self.mot_tracker.update(face_data)  # returns numpy array with bbox and id

            if len(track_bbs_ids) != 0:

                # Save cutout's of tracked persons into their buffer
                for tracked_person in range(len(track_bbs_ids)):

                    # Check whether the person already exists (i.e. has been detected before), and either return the old one, or create a new instance of Person
                    person_id = int(track_bbs_ids[tracked_person][4])
                    matched_person = [x for x in self.tracked_person_list if x.id == person_id]

                    if len(matched_person) == 0:
                        person = Person(person_id, self.buffer_size, self.object_lifetime)
                        self.tracked_person_list.append(person)
                    else:
                        person = matched_person[0]

                    bbox = normalise_bbox(track_bbs_ids[tracked_person][:4], image_dimensions)
                    # person_score = np.round(face_scores[tracked_person], decimals=3)
                    xP = int(bbox[0])
                    yP = int(bbox[1])
                    x1P = int(bbox[2])
                    y1P = int(bbox[3])

                    frame = self.orig_image[yP:y1P, xP:x1P]
                    frame = image_loader(frame)

                    # Reseting the age of each tracked Person Object
                    person.age = 0

                    # If a person was detected with a badge, draw a green box, if was detected to not have a badge - red, if it's still unknown - yellow and save image to BUFFER for further checks
                    if person.badge is None:
                        person.addImageToBuffer([frame])
                        color = (25, 25, 25)
                        score_label = "Evaluating..."
                    elif person.badge_number is not None and person.badge_number != 0:
                        score_label = 'Badge score: {}'.format(person.badge_score)
                        if person.badge_number == 1:
                            color = (61, 139, 255)
                        elif person.badge_number == 2:
                            color = (239, 131, 49)
                        elif person.badge_number == 3:
                            color = (6, 119, 17)
                        elif person.badge_number == 4:
                            color = (10, 6, 119)
                        elif person.badge_number == 5:
                            color = (225, 86, 220)
                    else:
                        score_label = "Not found!"
                        color = (255, 0, 0)

                    cv2.rectangle(self.orig_image, (xP, yP), (x1P, y1P), color, 2)
                    cv2.putText(self.orig_image, ('person id: {}'.format(person_id)), (xP, yP),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(self.orig_image, score_label, (xP, y1P), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            else:
                # Cover the scenario where there are people detected, but they couldn't be tracked
                pass
        else:
            # Cover the scanario where no people were detected - perhaps a "hibernation" mode approach (start checking only once every 3 seconds instead of every frame)
            pass

    def detect_badge(self, person_cutout, threshold=0.3, save_cutout=False):

        # print("detecting badge")
        scan_data = {'score': [], 'badge_cutout': []}
        badge_found = False

        with no_grad():
            badge_bbox_prediction = self.badge_predictor([image_loader(person_cutout)])

        for element in range(len(badge_bbox_prediction[0]["boxes"])):
            score = np.round(badge_bbox_prediction[0]["scores"][element].item(), decimals=2)
            print(score)
            if score >= threshold:
                badges = badge_bbox_prediction[0]["boxes"][element].cpu().numpy()
                xB = int(badges[0])
                yB = int(badges[1])
                x1B = int(badges[2])
                y1B = int(badges[3])
                person_cutout = tensor_to_image(person_cutout)
                badge_cutout = person_cutout[yB:y1B, xB:x1B]
                scan_data['score'].append(score)
                scan_data['badge_cutout'].append(badge_cutout)
                badge_found = True

        return badge_found, scan_data

    def evaluate_detected_badges(self, badge_score_list, threshold=None):
        # print("evaluating detected badge")
        if len(badge_score_list) > 0:
            confidence = sum(badge_score_list) / len(badge_score_list)

            if threshold is not None:
                if confidence < threshold:
                    return None

            return confidence

        return None

    def classify_badge(self, badge_cutout, threshold=0.8):
        # print("classifying badge")
        scan_data = {'score': [], 'label': []}
        badge_classified = False

        with no_grad():
            badge_class_prediction = self.badge_classifier([image_loader(badge_cutout, resize=False)])

        for element in range(len(badge_class_prediction[0]["labels"])):
            badge_score = np.round(badge_class_prediction[0]["scores"][element].item(), decimals=2)
            if badge_score >= threshold:
                scan_data['score'].append(badge_score)
                scan_data['label'].append(badge_class_prediction[0]['labels'][element].item())
                badge_classified = True

        return badge_classified, scan_data

    def evaluate_classified_badges(self, badge_class_list, class_score_list, threshold=None):
        # print("Evaluating classified badge")
        if len(class_score_list) > 0:
            badge_class_list = flatten_list(badge_class_list)
            class_score_list = flatten_list(class_score_list)

            predicted_badge_number = mode(badge_class_list)

            score_list = []
            for idx in range(len(badge_class_list)):
                if badge_class_list[idx] == predicted_badge_number:
                    score_list.append(class_score_list[idx])
            confidence = sum(score_list) / len(score_list)

            if threshold is not None:
                if confidence < threshold:
                    return None, confidence

            return predicted_badge_number, confidence
        return None, None

    def read_frame(self, sink):
        # Start FPS count engine
        if self.frame_id == 0:
            self.read_frames_fps.start()

        appsink_sample = GstApp.AppSink.pull_sample(sink)
        if appsink_sample is None:
            return False

        buff = appsink_sample.get_buffer()
        size, offset, _ = buff.get_sizes()
        frame_data = buff.extract_dup(offset, size)
        self.last_read_frame = {'scanned': False, 'image': bytes_to_array(frame_data)}

        self.read_frames_fps.update()
        self.read_frames_fps.stop()
        print("frame_reader fps: %s" % self.read_frames_fps.fps())

        return False  # Not necessary, but otherwise spams the terminal with an error message

    def start(self):
        pipe = pipeline.Pipe(self.id, self.cam_url)
        pipe.run()
        appsink = pipe.get_appsink()
        appsink.connect("new-sample", self.read_frame)
        print("Camera {} started".format(self.id))

    def stop(self):
        # pipeline.LOOP.quit()
        # TODO: disconnect the appsink
        self.last_read_frame = None

    def get_frame_bytes(self, read=True):
        while read:
            if self.last_read_frame is None:
                image = cv2.imread('app/interface/static/assets/img/offline.jpg')
                frame = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[1].tobytes()
            else:
                frame = cv2.imencode('.jpg', cv2.cvtColor(self.last_updated_frame, cv2.COLOR_BGR2RGB))[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def __del__(self):
        print("Camera {} turned off".format(self.id))
        '''if self.record is not None:
            print('Recording saved')
            self.out.release()'''
        '''if self.interface:
            self.cap.release()
            cv2.destroyWindow('Camera {}'.format(self.id))'''
        SurveillanceCamera.count -= 1


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
