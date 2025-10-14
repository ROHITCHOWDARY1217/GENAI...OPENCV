import cv2 as cv
import streamlit as st
import numpy as np
import time

st.title("Person Detection")
face_cascade = cv.CascadeClassifier("models/haarcascade_fullbody.xml")
video = cv.VideoCapture("personVideo/video1.mp4")
frame_placeholder = st.empty()

def personDetector():
    while True:
        isTrue, frame = video.read()
        if not isTrue:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        persons = face_cascade.detectMultiScale(gray, 1.1, 1)
        for (x, y, w, h) in persons:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if cv.waitKey(20) & 0xFF == ord("d"):
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        flip = cv.flip(frame, 1)
        
        frame_placeholder.image(flip)
    video.release()
    cv.destroyAllWindows()
    
if st.button("Start Person Detection"):
    personDetector()    
if st.button("Stop Person Detection"):
    video.release()
    cv.destroyAllWindows()