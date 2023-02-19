import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

def LSD_Detection(img):
    img = np.array(img)
    lsd = cv2.createLineSegmentDetector(0)
    dlines = lsd.detect(img)
    line_img = np.zeros(img.shape, dtype=np.float64) 
    try:
        for dline in dlines[0]:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            # print(x0, y0, x1, y1)
            cv2.line(line_img, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)
    except:
        pass
    return Image.fromarray(np.uint8(line_img)).convert("L")