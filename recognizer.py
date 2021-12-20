import cv2
import numpy as np
import os
print('''hello there
this is my face recogination app''')
inp=input("press Enter to proceed and press Q to quit ")
print("Look into the camera and wait")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
font=cv2.FONT_ITALIC
id =0
names = ['unknown', 'Mohan', 'Dinesh','gopika']
cam = cv2.VideoCapture(0)
while True:
    check, frame = cam.read()
    greyimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(greyimg, scaleFactor=1.05, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(greyimg[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        imageop=cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('imageop', frame)
    key = cv2.waitKey(100)
    if key == ord("q"):
        break
print("Exiting Program...")
cam.release()
cv2.destroyAllWindows()