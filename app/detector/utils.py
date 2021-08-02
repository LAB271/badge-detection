from PIL import Image
from torch import is_tensor
from torch.autograd import Variable
from torchvision import transforms
from cv2 import cvtColor, COLOR_RGB2BGR
from numpy import array

# loads an image and returns a tensor
# (automatically scales to required input size, therefore any image can be passed forward to the model)
resizeImg = transforms.Compose([transforms.Resize(300)])
imgToTensor = transforms.Compose([transforms.ToTensor()])

def image_loader(image, resize=True):
    # image = Image.open(image_name)
    if not is_tensor(image):
        if type(image) != 'PIL':
            image = Image.fromarray(image)
        image = imgToTensor(image).float()
    if resize:
        image = resizeImg(image).float()      
    
    image = Variable(image, requires_grad=True)
    return image

def tensor_to_image(tensor, cv2=False):
    image = transforms.ToPILImage()(tensor).convert("RGB")
    image = array(image)
    if cv2:
        image = cvtColor(image, COLOR_RGB2BGR)
    return image

# Make sure all bbox coordinates are inside the image
def normalise_bbox(bbox, image_dimensions):
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[2] > image_dimensions[1]:
        bbox[2] = image_dimensions[1]
    if bbox[3] > image_dimensions[0]:
        bbox[3] = image_dimensions[0]
    return bbox

def badge_num_to_color(idx):
    class_dict = {0:'not a', 1:'blue', 2:'orange', 3:'green', 4:'dark_blue', 5:'pink'}
    return class_dict[idx]

def flatten_list(list):
    return [item for sublist in list for item in sublist]

def print_alert(code, camera_id, person_id, det_conf = None, clas_conf = None):

    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    
    if code == 0:
        det_conf = "{}%".format(round(100 - det_conf*100), 2)
        clas_conf = "-" if clas_conf is None else "{}%".format(round(clas_conf), 2)
        print("-------------------------ALERT----------------------")
        print("Camera {} found that person {} does not have a badge".format(camera_id, person_id))
        print("DETECTOR confidence: {}  CLASSIFIER confidence: {}".format(det_conf, clas_conf))

    elif code == 1:
        print("-------------------------ALERT------------------------")
        print("Camera {} found that person {} is in a restricted area".format(camera_id, person_id))
        print("DETECTOR confidence: {}%  CLASSIFIER confidence: {}%".format(round(det_conf*100, 2), round(clas_conf*100, 2)))

    print("")
    print("")
    print("")
    print("")
    print("")
    print("")