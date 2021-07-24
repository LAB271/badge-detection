import os
from models.faceDetection.vision.ssd.config.fd_config import define_img_size
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

test_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class PersonDetector():

    def __init__(self, image_size=1280, threshold = 0.75, candidate_size = 1000):

        define_img_size(image_size)
        from models.faceDetection.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

        label_path = "models/faceDetection/voc-model-labels.txt"
        model_path = "models/faceDetection/version-RFB-320.pth"
        net_type = "RFB"
        class_names = ['BACKGROUND', 'face']
        num_classes = len(class_names)
        net = create_Mb_Tiny_RFB_fd(num_classes, is_test=True, device=test_device)
        self.model = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
        net.load(model_path)

        print('Person detection model loaded')

class BadgeDetector():

    def __init__(self, model_arch = 'mobilenet'):
        # resnet50      - inference time: 3.5s  - accuracy: very high
        # mobilenet_v3  - inference time: 0.2s  - accuracy: moderate

        if model_arch == 'mobilenet':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        elif model_arch == 'resnet50':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        self.model.load_state_dict(torch.load(os.path.join("models", "badgeDetection", model_arch), map_location=torch.device(test_device)))
        self.model.eval()
        
        print('Badge detection model loaded')