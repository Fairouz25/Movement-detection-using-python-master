import cv2
import numpy as np

video_path = r'C:\Users\komsi\Desktop\Smart Methods\Third Path\OpenCV example\test3.mov'
cap = cv2.VideoCapture(video_path) #read the video


while True:
    
    ret,frame1 = cap.read() # read the first frame
    ret,frame2 = cap.read() # read the second frame ( it a while True loop so it will read diffrenet frames each time)
    diff = cv2.absdiff(frame1,frame2) # a function to calculate the per-element absolute difference between two arrays
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY) # changing the 'diff' to gray scale
    blur = cv2.GaussianBlur(gray,(5,5), 0)
    _, thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh,None,iterations=3)
    contours, _ = cv2.findContours(dilated,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1500: # if the area of any object is less than 1500 then don't show it ( to avoid detecting small objects moving)
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2) # drawing a rectangle




    cv2.imshow('Movment detection',frame1)
    frame1 = frame2
    ret,frame2 = cap.read()
    if cv2.waitKey(40) == 27: #to close the video with ESC key (it's assigned as 27)
        break

cv2.destroyAllWindows()
cap.release()