from app.detector.utils import flatten_list, tensor_to_image


class Person(object):
    count = 0

    def __init__(self, id, max_buffer_size, max_lifetime=1):
        self.id = id
        self.max_buffer_size = max_buffer_size
        self.buffer = []    # Stores cutout of person
        '''
        Maybe introduce a buffer for badge_cutouts and then have the dictionary scan_data with score_det, score_clas, and label
        '''
        self.buffer_scanned = {'score_detection': [], 'badge_cutout': []} #, 'score_classification': [], 'label': []} # Stores detection scan data
        self.buffer_classified_badges = {'score_classification': [], 'label': []}
        self.badge = None   # Stores whether a badge was found, not found, or is still being looked for
        self.badge_number = None    # Stores the badge class
        self.badge_score = None     # Stores the detection, and later - classification model scores
        self.badge_check_count = 0
        self.max_lifetime = max_lifetime
        self.age = 0
        Person.count += 1

    def getBufferOppacity(self, value=''):
        if 'scan' in value:
            return len(flatten_list(self.buffer_scanned['score_detection']))
        elif 'person' in value:
            return len(self.buffer)
        else:
            raise ValueError

    def getBuffer(self, value=''):
        if 'scan' in value:
            if 'score' in value:
                return flatten_list(self.buffer_scanned['score_detection'])
            if 'badge' in value:
                return flatten_list(self.buffer_scanned['badge_cutout'])
            return self.buffer_scanned
        else:
            return self.buffer

    def clearBuffer(self, value=''):
        if 'scan' in value:
            self.buffer_scanned = {'score_detection': [], 'badge_cutout': []} #, 'score_classification': [], 'label': []}
        self.buffer = []
        self.max_buffer_size = 1

    # if formated is True, return a PIL image, if False, a pytorch tensor
    def getImage(self, idx, as_tensor=True, value=''):
        if 'scan' in value:
            image = flatten_list(self.buffer_scanned['badge_cutout'])[idx]
        else:
            image = self.buffer[idx]
            image = image[0]
        
        if not as_tensor:
            image = tensor_to_image(image, cv2=False)
        return image

    def addImageToBuffer(self, image):
        if self.getBufferOppacity('person') >= self.max_buffer_size:
            del self.buffer[0]
        self.buffer.append(image)

    def addScanDataToBuffer(self, scan_data):
        if self.getBufferOppacity('scan') >= self.max_buffer_size:
            del self.buffer_scanned['score_detection'][0]
            del self.buffer_scanned['badge_cutout'][0]
        self.buffer_scanned['score_detection'].append(scan_data['score'])
        self.buffer_scanned['badge_cutout'].append(scan_data['badge_cutout'])

    # Check whether the object is still being tracked
    def isAlive(self):
        self.age += 1
        if self.age >= self.max_lifetime:
            return False
        else:
            return True

    def __del__(self):
        # print("Deleting person {} from memory".format(self.get_id()))
        Person.count -= 1
