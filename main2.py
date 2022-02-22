from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from model import FacialExpressionModel

face_classifier = cv2.CascadeClassifier(r'C:\Users\Indovski\Desktop\ia_seminarska\haarcascade_frontalface_default.xml')
#classifier =load_model(r'C:\Users\Indovski\Desktop\ia_seminarska\model.h5')
model = FacialExpressionModel("model.json", "model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    labels = []
    gray_fr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_fr)

    for (x,y,w,h) in faces:

            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

       
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()