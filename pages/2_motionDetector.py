import cv2 as cv 
import numpy as np 
import streamlit as st 

st.title("Real-time Motion Detector")
face_detect = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
video = cv.VideoCapture(0)
frame_placeholder = st.empty()
def motionDetector():   
    prev_x = None 
    prev_y = None
    while True:
        isTrue, frame = video.read()
        if not isTrue:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if prev_x is not None and prev_y is not None:
                if abs(x - prev_x) > 5 or abs(y - prev_y) > 5: 
                    st.write("Motion Detected!")
                    video.release()
                    cv.destroyAllWindows()
                    return
            prev_x = x
            prev_y = y
        if cv.waitKey(20) & 0xFF == ord("d"):
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        flip = cv.flip(frame, 1)
        frame_placeholder.image(flip)      
    

if st.button("Start Motion Detection"):
    motionDetector()
if st.button("Stop Motion Detection"):
    video.release()
    cv.destroyAllWindows()
