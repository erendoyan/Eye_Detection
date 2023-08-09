import cv2 #Importing opencv
import numpy as np
import time #Importing time module

cap = cv2.VideoCapture(0) #Setting camera
eye = cv2.CascadeClassifier("haarcascade_eye (2).xml") #Setting haar_cascade for eye detection
last_eye_detected = time.time() #Defining a variable for time measuring

while True:
    ret,frame = cap.read() #Have frames
    frame = cv2.flip(frame,1) #Flipping frame because it takes from webcam
    frame = cv2.resize(frame,(640,480)) #Resizing frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Making frame gray for cascade
    eyes = eye.detectMultiScale(gray, 1.7, 8) #Detecting eyes
    if len(eyes)!=0: #If eyes is not none
        for (x, y, w, h) in eyes:
            last_eye_detected = time.time() #Defining last eye detected time
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) #Drawing rectangle to eyes
            cv2.putText(frame, 'Eye Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA) #Printing Eye Detected to screen
    else: #If it can not find eye
        if time.time() - last_eye_detected >=5: #If it can not find eye for 5 seconds or more
            cv2.putText(frame, 'Tired person detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA) #Printing Tired Person Detected
        else: #If it can not find eyes for 0-5 seconds
            cv2.putText(frame, "Eyes couldn't found!", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA) #Printing eye couldnt found


    cv2.imshow("Eye Detection",frame)
    if cv2.waitKey(20) & 0xFF == ord("q"): #q for exit
        break
cap.release()
cv2.destroyAllWindows()