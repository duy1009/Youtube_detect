import sys
sys.path.insert(0, './yolov7')
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import cv2
import torch


class Detect:
    def __init__(self, weights, img_size = 640, device ='cpu' ):
        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)  # check img_size


        if self.half:
            self.model.half()  # to FP16
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
    def precess_img(self, img, size):
        '''Resize image and convert to tensor 4D'''
        img = letterbox(img,size)[0]
        img_norm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_norm.astype(np.float32)/255
        img_norm = np.transpose(img_norm, (2, 0, 1))   
        return torch.from_numpy(img_norm).unsqueeze(0)
    def detect(self, image, conf_thres = 0.25, iou_thres=0.45, classes=None, agnostic_nms = False, augment =False):
        img = self.precess_img(image,(640,640))
        self.img_size_detect = img.shape
        pred = self.model(img, augment=augment)[0]  #detect
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        return pred
    def get_center(self, box):
        y = box.clone() if isinstance(box, torch.Tensor) else np.copy(box)
        y[:, 0] = (box[:, 0] + box[:, 2]) / 2  # x center
        y[:, 1] = (box[:, 1] + box[:, 3]) / 2  # y center
        return y
    def get_all_center(self, all_box):
        center = torch.tensor([])
        for i in all_box:
            center = np.append(center, self.get_center(i))
        

    def draw_all_box(self,img, pred):
        # Process detections
        for _, det in enumerate(pred):  # detections per image
            s, im0, ='', img
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.img_size_detect[2:], det[:, :4], im0.shape).round()
                # print(self.img_size_detect[2:])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
        return im0