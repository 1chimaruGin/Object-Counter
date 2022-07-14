import warnings
warnings.filterwarnings('ignore')

import cv2
import sys
import torch
import random
import argparse
import numpy as np

from trackutils.dataset import ReadInput, check_img_size
from trackutils.utils import non_max_suppression, preprocess, postprocess, plot_tracking

from tracker.byte_tracker import BYTETracker
from tracker.tracking_utils import Timer

sys.path.insert(0, "yolov7")
from yolov7.models.experimental import attempt_load
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import (
    select_device,
    time_synchronized,
    TracedModel,
)


class Detector(object):
    def __init__(self, model, device, imgsz, trace: bool = False) -> None:
        device = select_device(device)
        half = device.type != "cpu" and torch.cuda.is_available()

        # Load model
        model = attempt_load(model, map_location=device)  # Load FP32 model
        if trace:
            model = TracedModel(model)

        if half:
            model.half()  # to FP16

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        self.names = model.module.names if hasattr(model, "module") else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        if device.type != "cpu":
            model(
                torch.zeros(1, 3, *imgsz)
                .to(device)
                .type_as(next(model.parameters()))
            )  # run once

        self.half = half
        self.device = device
        self.model = model
        self.stride = stride
        self.imgsz = imgsz

    @torch.no_grad()
    def detect(self, img: np.ndarray, classes: int = 80, conf_thres: float = 0.001, nms_thresh: float = 0.7):
        """
        Detect objects in an image.
        :param img: image to detect objects in
        :param conf_thres: confidence threshold for object detection
        :param nms_thresh: nms threshold for non-maximum suppression
        :return: list of detected objects
        """
        img, _ = preprocess(img, self.imgsz)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        print("Before postprocess:", pred.shape)
        # output = postprocess(pred, classes, conf_thres, nms_thresh)
        output = non_max_suppression(pred, conf_thres, nms_thresh)
        print("Shape is", output[0].size())

        return output


class Tracker(object):
    def __init__(
        self,
        source,
        model,
        device='cuda:0',
        imgsz=640,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        aspect_ratio_thresh=1.6,
        min_box_area=10,
    ) -> None:
        self.detector = Detector(model, device, imgsz)
        self.tracker = BYTETracker(track_thresh, track_buffer, match_thresh)
        self.imgsz = check_img_size(imgsz)  # check img_size

        self.min_box_area = min_box_area
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.data = ReadInput(source)

    def track(self):
        results = []
        for frame_id, (img, path, cap) in enumerate(self.data):
            height, width = img.shape[:2]
            outputs = self.detector.detect(img)
            if outputs[0] is not None:
                online_targets = self.tracker.update(outputs[0], height, width, self.imgsz)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                online_im = plot_tracking(
                    img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=100,
                )
            else:
                online_im = img

            cv2.imshow("online", online_im)
            cv2.waitKey(1)
        
            
if __name__ == '__main__':
        tracker = Tracker('videos/palace.mp4', 'yolov7.pt', 'cuda', 640)
        tracker.track()