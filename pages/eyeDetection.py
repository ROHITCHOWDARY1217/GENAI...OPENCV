import cv2 as cv
import numpy as np
import streamlit as st

st.title("Eye Scanner")
face = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
eye = cv.CascadeClassifier("models/haarcascade_eye.xml")
video = cv.VideoCapture(0) 
frame_placeholder = st.empty()
def eyeScanner():

        while True:
            isTrue, frame = video.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            face_detect = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in face_detect:
                slice_gray = gray[y:y + h, x:x + w]
                slice_frame = frame[y:y + h, x:x + w]

                eye_detect = eye.detectMultiScale(slice_gray, scaleFactor=1.1, minNeighbors=5)
                for (ex, ey, ew, eh) in eye_detect:
                    cv.rectangle(slice_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            flip = cv.flip(frame, 1)
            frame_placeholder.image(flip)

            if cv.waitKey(1) & 0xFF == ord('d'):
                break

if st.button("Start Scanning"):
            eyeScanner()
if st.button("Stop Scanning"):
            video.release()
            cv.destroyAllWindows() 
