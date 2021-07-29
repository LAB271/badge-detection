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
from utils import normalise_bbox, image_loader, badge_num_to_color, print_alert
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

            # Look for a badge if it's not yet found
            if person.hasBadge() is None and person.getBufferOppacity() == person.getMaxBufferSize():
                
                det_confidence = self.detect_badges(person, 0.3, False)

            # if a person HAS a badge, but it's not yet known which one, initiate the classifier module
            if person.hasBadge() and person.badge_number is None or 0:
                
                clas_confidence = self.classify_badges(person, 0.8)
                
                if person.badge_score is None:
                    print('Failed to classify badge')

                elif person.badge_score > 0.8:
                    # Steps to take when the system is confident in the result of the badge detection models
                    if person.badge_number in self.allowed_badges:
                        person.clearBuffer('badges')
                    else:
                        person.setBadge(False)
                        print_alert(1, self.id, person.get_id(), det_confidence, person.badge_score)
                        # ALERT: this person is definitely not supposed to be here
                else:
                    # Steps to take when the system is NOT confident in the result of the badge detection models
                    # For now - check for the badge again
                    person.badgeCheckCount = 0
                    person.setBadge(None)

            
            # if the badge has been checked enough times and not found, report that badge was not found.
            if person.getBadgeCheckCount() == self.max_badge_check_count:
                person.badgeCheckCount += 1
                person.setBadge(False)
                print_alert(0, self.id, person.get_id(), det_confidence, person.badge_score)

                




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

    def detect_badges(self, person, threshold=0.3, save_cutouts=True):

        image_batch_tensor = person.getBuffer()
        score_list = []
        confidence = 0
        # Detection
        for image_id in range(person.getMaxBufferSize()):
            
            with no_grad():
                badge_bbox_prediction = self.badge_predictor(image_batch_tensor[image_id])
            
            person_cutout = person.getImage(image_id)
            for element in range(len(badge_bbox_prediction[0]["boxes"])):
                badge_score = np.round(badge_bbox_prediction[0]["scores"][element].item(), decimals=2)
                if badge_score >= threshold:
                    badges = badge_bbox_prediction[0]["boxes"][element].cpu().numpy()
                    xB = int(badges[0])
                    yB = int(badges[1])
                    x1B = int(badges[2])
                    y1B = int(badges[3])
                    badge_cutout = person_cutout[yB:y1B, xB:x1B]
                    score_list.append(badge_score)
                    # Saving image cutouts of badge and person with a badge for future retraining of detection model
                    if save_cutouts:
                        now = datetime.now()
                        current_time = now.strftime(r'%d-%m-%Y_%H-%M-%S')
                        rint = randint(1, 1000)
                        label = str(current_time) + str(rint)
                        path = os.path.join('output', 'badges', '{}.jpg'.format(label))
                        cv2.imwrite(path, badge_cutout)
                        path = os.path.join('output', 'person_cutouts', '{}.jpg'.format(label))
                        cv2.imwrite(path, person_cutout)

                    if self.interface:
                        cv2.rectangle(person_cutout, (xB, yB), (x1B, y1B), (0, 0, 255), 2)
                        cv2.putText(person_cutout, ('badge: ' + str(badge_score)), (xB, yB), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    badge_cutout = cv2.cvtColor(badge_cutout, cv2.COLOR_BGR2RGB)
                    person.addBadgeToBuffer([image_loader(badge_cutout, resize=False)])
        
        # Eval
        if len(score_list) > 0:
            confidence = sum(score_list)/len(score_list)
            person.badge = True
            person.clearBuffer()

        person.badgeCheckCount += 1

        return confidence

    def classify_badges(self, person, threshold = 0.9):

        image_batch_tensor = person.getBuffer('badges')
        badge_dict = {'labels': [], 'scores': []}
        confidence = 0
        
        # Detection 
        for image_id in range(person.getBufferOppacity('badges')):

            with no_grad():
                badge_class_prediction = self.badge_classifier(image_batch_tensor[image_id])

            for element in range(len(badge_class_prediction[0]["labels"])):
                badge_score = np.round(badge_class_prediction[0]["scores"][element].item(), decimals=2)
                if badge_score >= threshold:
                    badge_dict['labels'].append(badge_class_prediction[0]['labels'][element].item())
                    badge_dict['scores'].append(badge_score)
        
        # Eval
        if len(badge_dict['labels']) > 0:
            
            predicted_badge_number = mode(badge_dict['labels'])

            score_list=[]
            for idx in range(len(badge_dict['labels'])):
                if badge_dict['labels'][idx] == predicted_badge_number:
                    score_list.append(badge_dict['scores'][idx])
            confidence = sum(score_list)/len(score_list)

            person.badge_score = confidence
            person.badge_number = predicted_badge_number
 
        return confidence

    def __del__(self):
        print("Camera {} turned off".format(self.id))
        if self.record is not None:
            print('Recording saved')
            self.out.release()
        if self.interface:
            self.cap.release()
        cv2.destroyWindow('Camera {}'.format(self.id))
        SurveillanceCamera.count -= 1
