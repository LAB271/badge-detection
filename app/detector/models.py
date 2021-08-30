import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from app.models.faceDetection.vision.ssd.config.fd_config import define_img_size

test_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(test_device)
print(torch.version.cuda)


class PersonDetector:
    def __init__(self, net_type = 'RFB', image_size=640, candidate_size=1000):
        # RFB-640   - inference time: 0.09s     - accuracy: high
        # RFB-320   - inference time: 0.03s     - accuracy: lower
        # slim-320  - inference time: 0.03s     - accuracy: lower


        define_img_size(image_size)
        from app.models.faceDetection.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, \
            create_Mb_Tiny_RFB_fd_predictor
        from app.models.faceDetection.vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
        if net_type == 'slim':
            model_path = "app/models/faceDetection/version-slim-320.pth"
            # model_path = "models/pretrained/version-slim-640.pth"
            net = create_mb_tiny_fd(2, is_test=True, device=test_device)
            self.model = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
        elif net_type == 'RFB': 
            model_path = "app/models/faceDetection/version-RFB-640.pth"
            # model_path = "models/pretrained/version-RFB-640.pth"
            net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=test_device)
            self.model = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
        
        #class_names = ['BACKGROUND', 'face']
        net.load(model_path)
        print('Person detection model loaded')


class BadgeDetector:
    def __init__(self, model_arch='mobilenet_v2'):
        # resnet50      - inference time: 3.5s  - accuracy: very high
        # mobilenet_v3  - inference time: 0.2s  - accuracy: moderate

        if 'mobilenet' in model_arch:
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        elif model_arch == 'resnet50':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        self.model.load_state_dict(
            torch.load(os.path.join("app/models", "badgeDetection", model_arch), map_location=torch.device(test_device)))
        self.model.eval()

        print('Badge detection model loaded')

class BadgeClassifier:
    def __init__(self, model_arch='mobilenet_v2'):

        if 'mobilenet' in model_arch:
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=6)
        self.model.load_state_dict(
            torch.load(os.path.join("app/models", "badgeClassification", model_arch), map_location=torch.device(test_device)))
        self.model.eval()

        print('Badge classification model loaded - version SBP')
