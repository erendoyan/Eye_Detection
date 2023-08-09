import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
eye = cv2.CascadeClassifier("haarcascade_eye (2).xml")
last_eye_detected = time.time()

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye.detectMultiScale(gray, 1.7, 8)
    if len(eyes)!=0:
        for (x, y, w, h) in eyes:
            last_eye_detected = time.time()
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, 'Eye Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        if time.time() - last_eye_detected >=5:
            cv2.putText(frame, 'Tired person detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Eyes couldn't found!", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)


    cv2.imshow("Eye Detection",frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()