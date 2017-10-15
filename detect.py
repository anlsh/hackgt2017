from __future__ import print_function


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
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def detect_vid(cascade, filename):
    vid = cv2.VideoCapture()
    vid.open(filename)

    while vid.isOpened():
        (status, frame) = vid.read()
        detect_frame(cascade, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    vid.release()

def detect_picture(cascade, filename):

    img = cv2.imread(filename)
    detect_frame(cascade, img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        pass

def detect_frame(cascade, frame):

    print(frame.shape)
    scale = 1.0
    MAX_WIDTH = float(700)
    if frame.shape[0] > MAX_WIDTH:
        scale = MAX_WIDTH / frame.shape[0]

    frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    found = cascade.detectMultiScale(frame, minNeighbors=4, maxSize = (200, 200))

    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
            else:
                found_filtered.append(r)
    draw_detections(frame, found)
    draw_detections(frame, found_filtered, 3)
    print('%d (%d) found' % (len(found_filtered), len(found)))
    cv2.imshow('headhunter', frame)


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
