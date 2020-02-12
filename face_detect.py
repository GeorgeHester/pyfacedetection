import numpy as np
import cv2

faces_v = False
faces_2_v = False
faces_3_v = False
faces_4_v = False

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector_faces_2 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
detector_faces_3 = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
detector_faces_4 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
detector_bodys = cv2.CascadeClassifier('haarcascade_fullbody.xml')
detector_eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_upper_bodies = cv2.CascadeClassifier('haarcascade_upperbody.xml')
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_DUPLEX

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    faces_2 = detector_faces_2.detectMultiScale(gray, 1.3, 5)
    faces_3 = detector_faces_3.detectMultiScale(gray, 1.3, 5)
    faces_4 = detector_faces_4.detectMultiScale(gray, 1.3, 5)
    eyes = detector_eyes.detectMultiScale(gray , 5)

    for (x,y,w,h) in eyes:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img,(x,y+h+30),(x+w,y+h),(0,255,0), cv2.FILLED)
        cv2.putText(img,'EYES',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)
        cv2.rectangle(black, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(black,(x,y+h+30),(x+w,y+h),(0,255,0), cv2.FILLED)
        cv2.putText(black,'EYES',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)

    for (x,y,w,h) in faces:
        faces_v = True
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(img,(x,y+h+30),(x+w,y+h),(255,0,0), cv2.FILLED)
        cv2.putText(img,'FACE',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)
        cv2.rectangle(black,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(black,(x,y+h+30),(x+w,y+h),(255,0,0), cv2.FILLED)
        cv2.putText(black,'FACE',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)

    for (x,y,w,h) in faces_2:
        if faces_v == False:
            faces_2_v = True
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(x,y+h+30),(x+w,y+h),(255,0,0), cv2.FILLED)
            cv2.putText(img,'FACE_2',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)
            cv2.rectangle(black,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(black,(x,y+h+30),(x+w,y+h),(255,0,0), cv2.FILLED)
            cv2.putText(black,'FACE_2',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)

    for (x,y,w,h) in faces_3:
        if faces_v == False and faces_2_v == False:
            faces_3_v = True
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(x,y+h+30),(x+w,y+h),(255,0,0), cv2.FILLED)
            cv2.putText(img,'FACE_3',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)
            cv2.rectangle(black,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(black,(x,y+h+30),(x+w,y+h),(255,0,0), cv2.FILLED)
            cv2.putText(black,'FACE_3',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)

    for (x,y,w,h) in faces_4:
        if faces_v == False and faces_2_v == False and faces_3_v == False:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(x,y+h+30),(x+w,y+h),(255,0,0), cv2.FILLED)
            cv2.putText(img,'FACE_4',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)
            cv2.rectangle(black,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(black,(x,y+h+30),(x+w,y+h),(255,0,0), cv2.FILLED)
            cv2.putText(black,'FACE_4',(x+10,y+h+25),font,0.75,(255, 255, 255), 1)
        
    faces_v = False
    faces_2_v = False
    faces_3_v = False
    faces_4_v = False

    cv2.imshow('OUTPUT',img)
    cv2.imshow('DETECT', black)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
