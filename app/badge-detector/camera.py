import os
from datetime import datetime
from time import sleep
import cv2
import numpy as np
from numpy.core.fromnumeric import nonzero
from torch import no_grad
from random import randint
from person import Person
from sort.sort import Sort
from utils import normalise_bbox, image_loader, badge_num_to_color, print_alert, flatten_list, tensor_to_image
from statistics import mode


class SurveillanceCamera(object):
    count = 0

    def __init__(self, id, face_predictor, badge_predictor, badge_classifier, allowed_badges, path_to_stream, camera_fps, wanted_fps, buffer_size,
                 object_lifetime, max_badge_check_count, interface=True, record=None):

        self.id = id
        self.buffer_size = buffer_size
        self.object_lifetime = object_lifetime
        self.max_badge_check_count = max_badge_check_count
        self.interface = interface
        self.cap = cv2.VideoCapture(path_to_stream)
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.record = record

        if self.record is not None:
            now = datetime.now()
            current_time = now.strftime(r"%d-%m-%Y_%H-%M-%S")
            _, image = self.cap.read()
            #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.out = cv2.VideoWriter()
            self.out.open(os.path.join(self.record, 'Camera {} - {}.avi'.format(self.id, current_time)),
                                       fourcc, wanted_fps, (image.shape[1], image.shape[0]))

        self.mot_tracker = Sort(max_age=object_lifetime)
        self.face_predictor = face_predictor
        self.badge_predictor = badge_predictor
        self.badge_classifier = badge_classifier
        self.allowed_badges = allowed_badges
        self.tracked_person_list = []
        self.frame_id = 0
        SurveillanceCamera.count += 1
        self.frames_to_skip = int(camera_fps / wanted_fps)
        self.orig_image = None





    def update(self):
        ret, self.orig_image = self.cap.read()
        if ret is False:
            # Implement some sort of restart system here. Security cameras should not ever be turned off
            print("Exiting. Code 0")
            if self.record is not None:
                self.out.release()
            if self.interface:
                self.cap.release()
            sleep(10)
            return

        # FPS Control.
        self.frame_id += 1
        for i in range(2, self.frames_to_skip + 1):
            if self.frame_id % i == 0:
                # print("skipped frame {}".format(self.frame_id))
                return
        self.frame_id = 1

        image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)

        # Person Detection
        detected_faces, _, face_scores = self.face_predictor.predict(image, 500, 0.9) #(image, candidate_size/2, threshold)
        self.track_persons(detected_faces, face_scores)

        if self.interface:
            x = int(self.orig_image.shape[1] / 12)
            y = int(self.orig_image.shape[0] / 11)
            size = int(x / 40)
            cv2.putText(self.orig_image, "Tracking: {}".format(len(self.tracked_person_list)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (100, 0, 255), int(size))
            cv2.imshow('Camera {}'.format(self.id), self.orig_image)
            cv2.waitKey(1)

        if self.record is not None:
            self.out.write(self.orig_image)


        for person in self.tracked_person_list:
            
            # Self-destruction of Person objects (if they're not being used)
            if not person.isAlive():
                for index in range(len(self.tracked_person_list)):
                    if self.tracked_person_list[index] == person:
                        self.tracked_person_list.pop(index)
                        Person.count -= 1
                        break
                continue

            # Look for a badge if the person hasn't been found to have one yet
            if person.hasBadge() is None:
                person_cutout = person.getImage(len(person.buffer)-1, formated=False)
                scan_data = self.detect_badge(person_cutout)
                person.addScanDataToBuffer(scan_data) 

                # If there's enough data in the buffer to proceed, make a conclusion about the badges
                if person.getBufferOppacity('scan') == person.max_buffer_size:
                    
                    score_list = person.getBuffer('scanned scores')
                    detection_results = self.evaluate_detected_badges(score_list)

                    if detection_results is not None:
                        person.badge = True
                    else:
                        pass
                    person.badge_check_count+=1

            # if a person HAS a badge, but it's not yet known which one, initiate the classifier module
            if person.hasBadge() and person.badge_number is None or 0:
            
                badge_cutout = person.getImage(person.getBufferOppacity('scan')-1, formated=False, value='scan')
                label_list, score_list = self.classify_badge(badge_cutout, threshold=0.9)
                badge_number, confidence = self.evaluate_classified_badges(label_list, score_list)
                person.badge_number = badge_number
                person.badge_score = confidence
                
                if person.badge_score is None:
                    print('Failed to classify badge')

                elif person.badge_score > 0.8:
                    # Steps to take when the system is confident in the result of the badge detection models
                    if person.badge_number in self.allowed_badges:
                        person.clearBuffer()
                    else:
                        person.setBadge(False)
                        print_alert(1, self.id, person.get_id(), detection_results, person.badge_score)
                        # ALERT: this person is definitely not supposed to be here
                else:
                    # Steps to take when the system is NOT confident in the result of the badge detection models
                    # For now - check for the badge again
                    person.badge_check_count = 0
                    person.setBadge(None)

            
            # if the badge has been checked enough times and not found, report that badge was not found.
            if person.hasBadge() is None and person.badge_check_count == self.max_badge_check_count:
                person.badge_check_count += 1
                person.setBadge(False)
                print_alert(0, self.id, person.get_id(), detection_results, person.badge_score)

                




    def track_persons(self, detected_faces, face_scores):
        # Basic image prep
        #self.orig_image = cv2.flip(self.orig_image, 0)
        image_dimensions = self.orig_image.shape  # (h,w,c)
        image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
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
                    matched_person = [x for x in self.tracked_person_list if x.get_id() == person_id]

                    if len(matched_person) == 0:
                        person = Person(person_id, self.buffer_size, self.object_lifetime)
                        self.tracked_person_list.append(person)
                    else:
                        person = matched_person[0]

                    bbox = normalise_bbox(track_bbs_ids[tracked_person][:4], image_dimensions)
                    #person_score = np.round(face_scores[tracked_person], decimals=3)
                    xP = int(bbox[0])
                    yP = int(bbox[1])
                    x1P = int(bbox[2])
                    y1P = int(bbox[3])

                    frame = image[yP:y1P, xP:x1P]
                    frame = image_loader(frame)

                    # Reseting the age of each tracked Person Object
                    person.age = 0

                    # If a person was detected with a badge, draw a green box, if was detected to not have a badge - red, if it's still unknown - yellow and save image to BUFFER for further checks
                    if person.hasBadge() is None:
                        person.addImageToBuffer([frame])
                        color = (25, 25, 25)
                    elif person.badge_number is not None or 0:
                        if person.badge_number == 1:
                            color = (255, 139, 61)
                        elif person.badge_number == 2:
                            color = (49, 131, 239)
                        elif person.badge_number == 3:
                            color = (17, 119, 6)
                        elif person.badge_number == 4:
                            color = (119, 6, 10)
                        elif person.badge_number == 5:
                            color = (225, 86, 220)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(self.orig_image, (xP, yP), (x1P, y1P), color, 2)
                    cv2.putText(self.orig_image, ('person id: {}'.format(person_id)), (xP, yP), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(self.orig_image, ('badge score: {}'.format(person.badge_score)), (xP, y1P), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            else:
                # Cover the scenario where there are people detected, but they couldn't be tracked
                pass
        else:
            # Cover the scanario where no people were detected - perhaps a "hibernation" mode approach (start checking only once every 3 seconds instead of every frame)
            pass

    def detect_badge(self, person_cutout, threshold=0.4, save_cutout=False):

        scan_data = {'score': [], 'badge_cutout': []}

        with no_grad():
            badge_bbox_prediction = self.badge_predictor([image_loader(person_cutout)])

        for element in range(len(badge_bbox_prediction[0]["boxes"])):
            score = np.round(badge_bbox_prediction[0]["scores"][element].item(), decimals=2)
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
                
                # Saving image cutouts of badge and person with a badge for future retraining of detection model
                if save_cutout:
                    label = str(datetime.now().strftime(r'%d-%m-%Y_%H-%M-%S')) + str(randint(1, 1000))
                    path = os.path.join('output', 'badges', '{}.jpg'.format(label))
                    cv2.imwrite(path, badge_cutout)
                    path = os.path.join('output', 'person_cutouts', '{}.jpg'.format(label))
                    cv2.imwrite(path, person_cutout)

        return scan_data

    def evaluate_detected_badges(self, badge_score_list, threshold=None):

        if len(badge_score_list) > 0:
            confidence = sum(badge_score_list)/len(badge_score_list)

            # Optional: check whether the score is higher than threshold
            if threshold is not None:
                if confidence < threshold:
                    return None

            return confidence
        
        return None


    def classify_badge(self, badge_cutout, threshold=0.8):

        scan_data = {'label': [], 'score': []}

        with no_grad():
                badge_class_prediction = self.badge_classifier([image_loader(badge_cutout, resize=False)])

        for element in range(len(badge_class_prediction[0]["labels"])):
                badge_score = np.round(badge_class_prediction[0]["scores"][element].item(), decimals=2)
                if badge_score >= threshold:
                    scan_data['label'].append(badge_class_prediction[0]['labels'][element].item())
                    scan_data['score'].append(badge_score)

        return scan_data['label'], scan_data['score']


    def evaluate_classified_badges(self, badge_class_list, class_score_list, threshold=None):
        
        if len(class_score_list) > 0:
            
            predicted_badge_number = mode(badge_class_list)

            score_list = []
            for idx in range(len(badge_class_list)):
                if badge_class_list[idx] == predicted_badge_number:
                    score_list.append(class_score_list[idx])
            confidence = sum(score_list)/len(score_list)

            # Check whether the result is higher than a given threshold
            if threshold is not None:
                if confidence < threshold:
                    return None, None

            return predicted_badge_number, confidence

        return None, None
        

    def __del__(self):
        print("Camera {} turned off".format(self.id))
        if self.record is not None:
            print('Recording saved')
            self.out.release()
        if self.interface:
            self.cap.release()
        cv2.destroyWindow('Camera {}'.format(self.id))
        SurveillanceCamera.count -= 1
