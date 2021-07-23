from cv2 import cvtColor, COLOR_RGB2BGR
from numpy import array
from torchvision import transforms

class Person(object):

    count = 0
    def __init__(self, id, maxBufferSize, maxLifetime=1):
        self.id = id
        self.maxBufferSize = maxBufferSize
        self.buffer = []
        self.badge = None
        self.badgeCheckCount = 0
        self.maxLifetime = maxLifetime
        self.age = 0
        Person.count += 1

        #TODO: implement classification model functionality
        #self.badgeColor = None

    def getID(self):
        return self.id

    def getBufferOppacity(self):
        return len(self.buffer)

    def getBuffer(self):
        return self.buffer
    
    def clearBuffer(self):
        self.buffer = []

    def getMaxBufferSize(self):
        return self.maxBufferSize

    # if formated is True, return a PIL image, if False, a pytorch tensor
    def getImage(self, idx, formated=True):
        image = self.buffer[idx]
        image = image[0]
        if formated:
            image = transforms.ToPILImage()(image).convert("RGB")
            image = cvtColor(array(image), COLOR_RGB2BGR)
        return image

    def addImageToBuffer(self, image):
        if self.getBufferOppacity() >= self.getMaxBufferSize():
            del self.buffer[0]
        self.buffer.append(image)
    
    def hasBadge(self):
        return self.badge
    
    def setBadge(self, value=None):
        if value != True:
            self.badgeCheckCount +=1
        self.badge = value

    # Check whether the object is still being tracked
    def isAlive(self):
        self.age += 1
        if self.age >= self.maxLifetime:
            return False
        else:
            return True

    def getBadgeCheckCount(self):
        return self.badgeCheckCount

    def __del__(self):
        #print("Deleting person {} from memory".format(self.getID()))
        pass