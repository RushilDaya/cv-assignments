# this is the final code version
import numpy as np 
import cv2 
import functions as fc

outputStream = fc.createOutput('output.avi')
'''
# part 1 - sobel and canny filters
#---------------------------------------------------

video = fc.openVideo('source_videos/part1.mp4') 

fc.baseVideo(video,  outputStream ,5)
fc.sobelVideo(video, outputStream ,5)
fc.cannyVideo(video, outputStream ,5)

# part 2 - hough transform section
#---------------------------------------------------
video = fc.openVideo('source_videos/part2.mp4')
fc.houghCircleVideo(video, outputStream, 5)
'''
# part 3 - object detection on stills
#---------------------------------------------------
'''
templateImage = fc.openImage('source_videos/part3_object.jpg')
sceneImage = fc.openImage('source_videos/part3_base.jpg')
secondSceneImage = fc.openImage('source_videos/part3_second.jpg')
fc.doSift(templateImage, sceneImage, outputStream ,2, 3)
fc.doSift(templateImage, secondSceneImage, outputStream, 2, 3)
'''
# part 4 -object detection on moving object
#---------------------------------------------------

video = fc.openVideo('source_videos/part4.mp4')
templateImage = fc.openImage('source_videos/part3_object.jpg')
fc.doSiftVideo(templateImage, video, outputStream, 5)

# part 5 - special fun section
#---------------------------------------------------



# release -
#---------------------------------------------------
#video.release()
outputStream.release()
cv2.destroyAllWindows()