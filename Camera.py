import cv2
import time
from torch import no_grad
import numpy as np
from sort.sort import *
from Person import Person
from utils import *

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (1920,1080))


class SurveillanceCamera(object):

    count = 0
    def __init__(self, id_number, face_predictor, badge_predictor, path_to_stream, buffer_size, object_lifetime, max_badge_check_count, interface=True):

        self.id = id_number 
        self.buffer_size = buffer_size
        self.object_lifetime = object_lifetime
        self.max_badge_check_count = max_badge_check_count
        self.interface = interface
        self.cap = cv2.VideoCapture(path_to_stream) 
        self.mot_tracker = Sort(max_age=object_lifetime) 
        self.face_predictor = face_predictor
        self.badge_predictor = badge_predictor
        self.tracked_person_list = []
        self.frame_id = 0
        SurveillanceCamera.count += 1


    def update(self):

            ret, self.orig_image = self.cap.read()
            self.frame_id += 1
            if ret is False:
                # Implement some sort of restart system here. Security cameras should not ever be turned off
                print("Exiting. Code 0")
                return False

            # Basic image prep
            image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
            image_dimensions = self.orig_image.shape     # (h,w,c)

            # Person Detection
            faces, _, face_scores = self.face_predictor.predict(image, 500, 0.75)

            # If any persons were detected, track them, and add cutout images to their BUFFER
            if len(faces) != 0:
                # Formating the arrays for (deep)SORT into a numpy array that contains lists of (x1,y1,x2,y2,score)
                face_data = []
                for i in range(len(faces)):
                    # Changing the coordinates to bound the body instead of the face but also ensuring it doesn't go outside of the image bounds
                    # Head to body ratio is ~ 1:4 - 1:8. That can be used to mark the required body size knowing the head measurements
                    ratioW = faces[i][2]-faces[i][0]
                    ratioH = (faces[i][3]-faces[i][1])*5
                    faces[i][0] = int(faces[i][0])-ratioW
                    faces[i][1] = int(faces[i][1])
                    faces[i][2] = int(faces[i][2])+ratioW
                    faces[i][3] = faces[i][1] + ratioH
                    faces[i] = normaliseBBox(faces[i], image_dimensions)
                    temp = np.append(faces[i], face_scores[i])
                    face_data.append(temp)
                face_data = np.array(face_data)
    
                # Calling the person tracker
                track_bbs_ids = self.mot_tracker.update(face_data) #returns numpy array with bbox and id
                
                if len(track_bbs_ids) != 0:

                    # Save cutout's of tracked persons into their buffer
                    for tracked_person in range(len(track_bbs_ids)):

                        # Check whether the person already exists (i.e. has been detected before), and either return the old one, or create a new instance of Person
                        person_id = int(track_bbs_ids[tracked_person][4])
                        if len(self.tracked_person_list) != 0: 
                            for index in range(len(self.tracked_person_list)):
                                tracked_person_id = self.tracked_person_list[index].getID()
                                if tracked_person_id == person_id:
                                    #print("match found. ID: {}".format(person_id))
                                    for index in range(len(self.tracked_person_list)):
                                        if self.tracked_person_list[index].getID() == person_id:
                                            person = self.tracked_person_list[index]
                                            break
                                    break
                                elif index == len(self.tracked_person_list)-1:
                                    #print("Creating new instance with ID: {}".format(person_id))
                                    person = Person(person_id, self.buffer_size, self.object_lifetime)
                                    self.tracked_person_list.append(person)
                                    break
                        else:
                            #print("Creating new instance with ID: {}".format(person_id))
                            person = Person(person_id, self.buffer_size, self.object_lifetime)
                            self.tracked_person_list.append(person)                       
                        
                        bbox = normaliseBBox(track_bbs_ids[tracked_person][:4], image_dimensions)
                        person_score = np.round(face_scores[tracked_person], decimals=3)
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
                            color = (255, 255, 0)
                        elif person.hasBadge() is True:
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)

                        cv2.rectangle(self.orig_image, (xP, yP), (x1P, y1P), color, 2)
                        cv2.putText(self.orig_image, ('person {} - {}'.format(person_id, person_score)), (xP, yP), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    # Cover the scenario where there are people detected, but they couldn't be tracked
                    pass
            else:
                # Cover the scanario where no people were detected - perhaps a "hibernation" mode approach (start checking only once every 3 seconds instead of every frame)
                print("Camera {} is now in power saving mode".format(self.id))
                time.sleep(3)

            if self.interface:
                x = int(self.orig_image.shape[1]/11)
                y = int(self.orig_image.shape[0]/11)
                cv2.putText(self.orig_image, "Tracking: {}".format(len(self.tracked_person_list)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (100,0,255), 8)
                cv2.imshow('Camera {}'.format(self.id), self.orig_image)
                # QUESTION
                # added a waitkey here because otherwise the visual interface freezes after a badge is found. But why? When a badge is found that's exactly when there's supposed to be less computation going on, i.e. the program should run smoother
                cv2.waitKey(1)


            # Check the buffer size for each person, if available, check for their badges
            for person in self.tracked_person_list:

                # Self-destruction of Person objects (if they're not being used)
                if not person.isAlive():
                    for index in range(len(self.tracked_person_list)):
                        if self.tracked_person_list[index] == person:
                            self.tracked_person_list.pop(index)
                            Person.count -= 1
                            break
                    continue

                if person.hasBadge():
                    # Exit the loop
                    continue

                if person.hasBadge() == None and person.getBufferOppacity() == self.buffer_size:
                    image_batch = person.getBuffer()
                    badge_list = []

                    # Badge Detection
                    for image_id in range(self.buffer_size):
                        
                        with no_grad():
                            badge_prediction = self.badge_predictor(image_batch[image_id])

                        cutout_image = person.getImage(image_id)
                        for element in range(len(badge_prediction[0]["boxes"])):
                            badge_score = np.round(badge_prediction[0]["scores"][element].cpu().numpy(), decimals= 2)
                            if badge_score > 0.9:  
                                badges = badge_prediction[0]["boxes"][element].cpu().numpy()
                                badges = badges/(self.orig_image.shape[0]/cutout_image.shape[0])
                                xB = int(badges[0])# + xP - 10
                                yB = int(badges[1])# + yP - 10
                                x1B = int(badges[2])# + xP + 10
                                y1B = int(badges[3]) #+ yP + 10
                                badge_list.append(badge_score)
                                cv2.rectangle(cutout_image, (xB, yB), (x1B, y1B), (0,0,255), 2)
                                cv2.putText(cutout_image, ('badge: ' + str(badge_score)), (xB, yB), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Demo-ing the BUFFER
                        #windowName = 'person {}'.format(person.getID())
                        #cv2.imshow(windowName, cutout_image)
                        #cv2.waitKey(1)
                    #cv2.destroyWindow(windowName)
                    
                    # Badge Evaluation
                    if len(badge_list) > 0:
                        confidence = sum(badge_list)/len(badge_list)
                        if confidence >= 0.90:
                            # 
                            # implement badge classification here
                            #
                            value = True
                            person.clearBuffer() #This person won't be checked again - free-ing up memory
                            #print("Person {} is wearing a SBP badge. Confidence: {}".format(person.getID(), np.round(confidence, decimals=2)))
                        elif confidence >=0.60:
                            value = None
                            #print("Can't distinguish whether person {} is wearing a badge. Checking again".format(person.getID()))
                        else:
                            value = None
                            #print("Person {} is not wearing a SBP badge".format(person.getID()))
                        person.setBadge(value)
                    else:
                        person.setBadge(None)
                        #print("Person {} is not wearing a SBP badge".format(person.getID()))
                    
                    # if the badge has been checked enough times and not found, report that badge was not found.
                    if person.getBadgeCheckCount() == self.max_badge_check_count:
                        print("ALERT")
                        print("Camera {} found that person {} does not have a badge".format(self.id, person.getID()))
                        person.setBadge(False)
        

    def __del__(self):
        self.cap.release()
        cv2.destroyWindow('Camera {}'.format(self.id))
        SurveillanceCamera.count -= 1