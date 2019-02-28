# The following tutorial was used as a starting point for the assignment as it gives insight in how to read video files
# https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html?fbclid=IwAR0LYIh0gPbtVyeMiD6eRXfwCKR4eDNXF_ANPxrEi_2Ewioe87GU3GErgxE

import numpy as np 
import cv2
import manipulations as mp

FRAME_RATE = 60.0
RESOLUTION = (1280, 720)
INPUT_FILE_NAME = 'input-2.mp4'
OUTPUT_FILE_NAME = 'output-file.avi'
EFFECTS = {
    0 :  {'name':'grayscale','len':5},
    5 :  {'name':'grayscale-smoothing','len':5, 'kernel_size':51, 'variance_start_value':1,'variance_end_value':51},
    10 : {'name':'grayscale-edge-preserve','len':5,'min_applications':1, 'max_applications':5},
    15 : {'name':'color','len':5},
    20 : {'name':'grab-object-rgb','len':5},
    25 : {'name':'grab-object-hsv','len':5},
    30 : {'name':'grab-object-rgb-morp','len':5},
    35 : {'name':'face-detect','len':5},
    40 : {'name':'face-blur','len':5},
    45 : {'name':'laplace-filter','len':15}
}

# define i/o objects 
cap = cv2.VideoCapture(INPUT_FILE_NAME)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter(OUTPUT_FILE_NAME,fourcc, FRAME_RATE, RESOLUTION,1 )

frame_number = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        [method, argsObj] = mp.getMethod(frame_number, FRAME_RATE, EFFECTS)
        modified  = method(frame,argsObj)
        out.write(modified)
        cv2.imshow('frame', modified)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):
            while True:
                if  cv2.waitKey(1) & 0xFF == ord('c'):
                    break
        frame_number +=1 
    else:
        break
        
# release objects    
cap.release()
out.release()
cv2.destroyAllWindows()