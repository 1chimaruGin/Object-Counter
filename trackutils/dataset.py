import os
import cv2
import math
import glob
import numpy as np
from torch.backends import cudnn
from pathlib import Path

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]


def setup_cudnn() -> None:
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = True
    cudnn.deterministic = False


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))

    return (new_size, new_size)


class ReadInput:
    def __init__(self, path):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise FileNotFoundError(f"Invalid path {p}")

        imgp = [i for i in files if i.split(".")[-1] in IMG_FORMATS]
        vidp = [v for v in files if v.split(".")[-1] in VID_FORMATS]
        self.files = imgp + vidp
        self.nf = len(self.files)
        self.type = "image"
        if any(vidp):
            self.add_video(vidp[0])  # new video
        else:
            self.cap = None

    @staticmethod
    def checkext(path):
        file_type = "image" if path.split(".")[-1].lower() in IMG_FORMATS else "video"
        return file_type

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.checkext(path) == "video":
            self.type = "video"
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.add_video(path)
                ret_val, img = self.cap.read()
        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR

        return img, path, self.cap

    def add_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
  