# The following tutorial was used as a starting point for the assignment as it gives insight in how to read video files
# https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html?fbclid=IwAR0LYIh0gPbtVyeMiD6eRXfwCKR4eDNXF_ANPxrEi_2Ewioe87GU3GErgxE

import numpy as np 
import cv2
import manipulations as mp

FRAME_RATE = 60.0
RESOLUTION = (1280, 720)

EFFECTS = {
    0 :  {'name':'sobel','len':5, 'kernel_start_size':3, 'kernel_end_size':15}
}

# define i/o objects 
cap = cv2.VideoCapture(0)
frame_number = 0
while(True):
    ret, frame = cap.read()
    [method, argsObj] = mp.getMethod(frame_number, FRAME_RATE, EFFECTS)
    modified  = method(frame,argsObj)
    cv2.imshow('frame', modified)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        while True:
            if  cv2.waitKey(1) & 0xFF == ord('c'):
                break
    frame_number +=1 
        
# release objects    
cap.release()