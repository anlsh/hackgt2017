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

def _dist(box1, box2):
    c1 = _center(box1)
    c2 = _center(box2)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def _center(box):
    return (box[0] + box[2]/2, box[1] + box[3]/2)

class Rectifier():

    def __init__(self):
        self.MEMORY_DEPTH = 2
        self.cell_size = 500
        self.old_boxes = dq()

    def package(self, boxes):
        cells = {}
        for b in boxes:
            b = tuple(b)
            cell = self.getcell(b)
            if cells.get(cell, None) is None:
                cells[cell] = list()
                cells[cell].append(b)
            else:
                cells[cell].append(b)

        return cells

    def nearcells(self, cell):
        """
        Return the cell-coords of the 3x3 grid including current cell
        """
        (x, y) = cell
        return [(x-1, y+1), (x-1, y), (x-1, y-1),
                (x, y+1), (x, y), (x, y-1),
                (x+1, y+1), (x+1, y), (x+1, y-1)]

    def getcell(self, box):
        """Return cell coords of the center of the box"""
        cent = _center(box)
        return (int(cent[0] / self.cell_size), int(cent[1] / self.cell_size))

    def filter(self, boxes):
        toRet = []
        boxmap = self.package(boxes)
        if len(self.old_boxes) <= self.MEMORY_DEPTH:
            self.old_boxes.appendleft(boxmap)
            return boxes
        else:
            for b in boxes:
                leap = 0
                bcell = self.getcell(b)
                for bmap in self.old_boxes:
                    minor_leap = 10000000000000
                    for ncell in self.nearcells(bcell):
                        cboxes = bmap.get(ncell, None)
                        if not cboxes:
                            cboxes = []
                        for c in cboxes:
                            minor_leap = min(minor_leap, _dist(b, c))
                    leap = max(minor_leap, leap)

                if leap <= 2*self.cell_size:
                    toRet.append(b)

            self.old_boxes.pop()
            self.old_boxes.appendleft(boxmap)

            return toRet

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
