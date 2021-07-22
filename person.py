from cv2 import cvtColor, COLOR_RGB2BGR
from numpy import array
from torchvision import transforms

class Person(object):

    def __init__(self, id, maxBufferSize, lifetime=10):
        self.id = id
        self.maxBufferSize = maxBufferSize
        self.buffer = []
        self.badge = False
        self.lifetime = lifetime

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
            print("max buffer reached")
            del self.buffer[0]
        self.buffer.append(image)
    
    def hasBadge(self):
        return self.badge
    
    def setBadge(self, value=True):
        self.badge = value

    
            
