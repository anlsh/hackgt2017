from __future__ import print_function
from collections import deque as dq


"""
'''
example to detect upright people in images using HOG features
Usage:
    peopledetect.py <image_names>
Press any key to continue, ESC to stop.
'''
"""
import time
import sys
import numpy as np
import cv2

# https://github.com/opencv/opencv/blob/master/samples/python/peopledetect.py
def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger boxes than the real objects.
        # so we slightly shrink the boxes to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def detect_vid(cascade, filename, smoothlist=False):
    vid = cv2.VideoCapture()
    vid.open(filename)

    rectifier = Rectifier()

    while vid.isOpened():
        for i in range(5):
            (status, frame) = vid.read()
        detect_frame(cascade, frame, rectifier)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    vid.release()

def detect_picture(cascade, filename):

    img = cv2.imread(filename)
    detect_frame(cascade, img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        pass

def detect_frame(cascade, frame, rectifier=None):

    print(frame.shape)
    scale = 1.0
    MAX_WIDTH = float(700)
    if frame.shape[0] > MAX_WIDTH:
        scale = MAX_WIDTH / frame.shape[0]

    #frame = frame[int(frame.shape[0]/2), ]
    frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    found = cascade.detectMultiScale(frame, minNeighbors=2, maxSize = (200, 200))
    if rectifier:
        found = rectifier.filter(found)

    draw_detections(frame, found, 2)
    # draw_detections(frame, found_filtered, 3)
    # print('%d (%d) found' % (len(found_filtered), len(found)))
    cv2.imshow('headhunter', frame)

<<<<<<< HEAD
=======
def _dist(box1, box2):
    c1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
    c2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

class Rectifier():

    def __init__(self):
        self.MEMORY_DEPTH = 1
        self.MAX_JUMP = 450
        self.cell_size = 50
        self.old_boxes = dq()

    def package(boxes):
        cells = {}
        for b in boxes:
            cellx = int(b[0] / cell_size)
            celly = int(b[1] / cell_size)
            if cells[(cellx, celly)] is None:
                cells[(cellx, celly)] = list(b)
            else:
                cells[(cellx, celly)].append(b)

        return cells


    def filter(self, boxes):
        toRet = []
        if len(self.old_boxes) <= self.MEMORY_DEPTH:
            self.old_boxes.appendleft(boxes)
            return boxes
        else:
            for b in boxes:
                leap = max([min([_dist(b, o) for o in oset]) for oset in self.old_boxes])
                if leap <= self.MAX_JUMP:
                    toRet.append(b)
            self.old_boxes.pop()
            self.old_boxes.appendleft(boxes)

            return toRet


>>>>>>> 5eae08438bb00309d72089c1d6f75eefe6952f7b

def get_cascade():

    cascade = cv2.CascadeClassifier()
    cascade.load("cascades/fullbody_good.xml")
    return cascade


if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it

#   print(sys.argv)
    cascade = get_cascade()
    detector = detect_vid if sys.argv[1] == "v" else detect_picture
    detector(cascade, sys.argv[2])
