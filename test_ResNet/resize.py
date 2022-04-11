import cv2
import numpy as np


def resize(rawimg):
    fx = 28.0 / rawimg.shape[0]
    fy = 28.0 / rawimg.shape[1]
    fx = fy = min(fx, fy)
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    outimg = np.ones((28, 28), dtype=np.uint8) * 255
    w = img.shape[1]
    h = img.shape[0]
    x = round((28 - w) / 2)
    y = round((28 - h) / 2)
    outimg[y:y + h, x:x + w] = img
    return outimg
