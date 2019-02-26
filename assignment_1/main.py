# The following tutorial was used as a starting point for the assignment as it gives insight in how to read video files
# https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html?fbclid=IwAR0LYIh0gPbtVyeMiD6eRXfwCKR4eDNXF_ANPxrEi_2Ewioe87GU3GErgxE

import numpy as np 
import cv2
import manipulations as mp

FRAME_RATE = 60.0
RESOLUTION = (1280, 720)
INPUT_FILE_NAME = 'input-file.mp4'
OUTPUT_FILE_NAME = 'output-file.avi'
EFFECTS = {
    0 :  {'name':'grayscale','len':5},
    5 :  {'name':'grayscale-smoothing','len':5, 'kernel_start_size':3,'kernel_end_size':21},
    10 : {'name':'grayscale-edge-preserve','len':5},
    15 : {'name':'color','len':5},
    20 : {'name':'grab-object-rgb','len':5},
    25 : {'name':'grab-object-hsv','len':5},
    30 : {'name':'grab-object-morph','len':5},
    35 : {'name':'creative','len':25}
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

        frame_number +=1 
    else:
        break
        
# release objects    
cap.release()
out.release()
cv2.destroyAllWindows()