# The following tutorial was used as a starting point for the assignment as it gives insight in how to read video files
# https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html?fbclid=IwAR0LYIh0gPbtVyeMiD6eRXfwCKR4eDNXF_ANPxrEi_2Ewioe87GU3GErgxE

import numpy as np 
import cv2
import manipulations as mp
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

FRAME_RATE = 60.0
RESOLUTION = (1280, 720)

EFFECTS = {
    0: {'name':'hough-circle', 'len':5, 'lower_threshold_start':10,'threshold_gap':100},
    500 : {'name':'sobel','len':5, 'kernel_start_size':3, 'kernel_end_size':15}
}

## sift train on the image of money

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread('trainImage.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()

# define i/o objects 
cap = cv2.VideoCapture(0)
frame_number = 0

while(True):
    ret, frame = cap.read()
    image = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = image.copy()

	# detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
	# draw the original bounding boxes
    for (x, y, w, h) in rects:
	    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# sho the output images
    cv2.imshow("After NMS", image)
    # [method, argsObj] = mp.getMethod(frame_number, FRAME_RATE, EFFECTS)
    # modified  = method(frame,argsObj)
    # cv2.imshow('frame', modified)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if cv2.waitKey(1) & 0xFF == ord('p'):
    #     while True:
    #         if  cv2.waitKey(1) & 0xFF == ord('c'):
    #             break
    # frame_number +=1 
        
# release objects    
cap.release()