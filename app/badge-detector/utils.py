from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

# loads an image and returns a tensor
# (automatically scales to required input size, therefore any image can be passed forward to the model)
loaderResize = transforms.Compose([transforms.Resize(300), transforms.ToTensor()])
loader = transforms.Compose([transforms.ToTensor()])

def image_loader(image, resize=True):
    # image = Image.open(image_name)
    if type(image) != 'PIL':
        image = Image.fromarray(image)

    if resize:
        image = loaderResize(image).float()
    else:
        image = loader(image).float()
    image = Variable(image, requires_grad=True)
    return image


# Make sure all bbox coordinates are inside the image
def normalise_bbox(bbox, image_dimensions):
    if bbox[0] < 0:
        bbox[0] = 1
    if bbox[1] < 0:
        bbox[0] = 1
    if bbox[2] > image_dimensions[1]:
        bbox[2] = image_dimensions[1]-1
    if bbox[3] > image_dimensions[0]:
        bbox[3] = image_dimensions[0]-1
    return bbox

def badge_num_to_color(idx):
    class_dict = {0:'not a', 1:'blue', 2:'orange', 3:'green', 4:'dark_blue', 5:'pink'}
    return class_dict[idx]


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