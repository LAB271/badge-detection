from cv2 import cvtColor, COLOR_RGB2BGR
from numpy import array
from torchvision import transforms
from app.detector.utils import flatten_list


class Person(object):
    count = 0

    def __init__(self, id, max_buffer_size, max_lifetime=1):
        self.id = id
        self.max_buffer_size = max_buffer_size
        self.buffer = []
        self.buffer_badges = []
        self.buffer_scanned = {'score': [], 'badge_cutout': []}
        self.badge = None
        self.badge_number = None
        self.badge_score = None
        self.badge_check_count = 0
        self.max_lifetime = max_lifetime
        self.age = 0
        Person.count += 1

        # TODO: implement classification model functionality
        # self.badgeColor = None

    def get_id(self):
        return self.id

    def getBufferOppacity(self, value=''):
        if 'badge' in value:
            return len(self.buffer_badges)
        elif 'scan' in value:
            return len(flatten_list(self.buffer_scanned['score']))
        else:
            return len(self.buffer)

    def getBuffer(self, value=''):
        if 'badge' in value:
            return self.buffer_badges
        elif 'scan' in value:
            if 'score' in value:
                return flatten_list(self.buffer_scanned['score'])
            if 'image' or 'badge' in value:
                return flatten_list(self.buffer_scanned['badge_cutout'])
            return self.buffer_scanned
        else:
            return self.buffer

    def clearBuffer(self, value=''):
        if 'badge' in value:
            self.buffer_badges = []
        elif 'scan' in value:
            self.buffer_scanned = {'score': [], 'badge_cutout': []}
        self.buffer = []
        self.max_buffer_size = 1

    # if formated is True, return a PIL image, if False, a pytorch tensor
    def getImage(self, idx, formated=True, value=''):
        if 'badge' in value:
            image = self.buffer_badges[idx]
            image = image[0]
        if 'scan' in value:
            image = flatten_list(self.buffer_scanned['badge_cutout'])[idx]
        else:
            image = self.buffer[idx]
            image = image[0]
        
        if formated:
            image = transforms.ToPILImage()(image).convert("RGB")
            image = cvtColor(array(image), COLOR_RGB2BGR)
        return image

    def addImageToBuffer(self, image):
        if self.getBufferOppacity() >= self.max_buffer_size:
            del self.buffer[0]
        self.buffer.append(image)

    def addBadgeToBuffer(self, image):
        self.buffer_badges.append(image)

    def addScanDataToBuffer(self, scan_data):
        if self.getBufferOppacity('scan') >= self.max_buffer_size:
            del self.buffer_scanned['score'][0]
            del self.buffer_scanned['badge_cutout'][0]
        self.buffer_scanned['score'].append(scan_data['score'])
        self.buffer_scanned['badge_cutout'].append(scan_data['badge_cutout'])

    def hasBadge(self):
        return self.badge

    def setBadge(self, value=None, badge_number=None):
        if value is None:
            self.badge_check_count += 1
        if badge_number is not None:
            self.badge_number = badge_number
        self.badge = value

    # Check whether the object is still being tracked
    def isAlive(self):
        self.age += 1
        if self.age >= self.max_lifetime:
            return False
        else:
            return True

    def __del__(self):
        # print("Deleting person {} from memory".format(self.get_id()))
        pass
