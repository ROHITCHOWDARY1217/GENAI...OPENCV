import cv2 as cv
import numpy as np
import streamlit as st

st.title("Face Scanner")

face = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")

video = cv.VideoCapture(0)

scan_video = cv.VideoCapture("animations/Scan Matrix.mov")
animation_video = cv.VideoCapture("animations/Apple Face ID.mp4")
frame_placeholder = st.empty()
frame_placeholder2 = st.empty()

def faceScanner():
    while True:
        isTrue, frame = video.read()
        isScan, scan_frame = scan_video.read()
        isAnim, anim_frame = animation_video.read()
        frame_count += 1
        if frame_count % 3 != 0:
            continue


        if not isTrue:
            break
        if not isScan:
            scan_video.set(cv.CAP_PROP_POS_FRAMES, 0)
            isScan, scan_frame = scan_video.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            scan_frame_resized = cv.resize(scan_frame, (w, h))
            frame[y:y + h, x:x + w] = blend_overlay(frame[y:y + h, x:x + w], scan_frame_resized)

        frame_rgb = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
        if isAnim:
            frame_placeholder2.image(anim_frame)
        frame_placeholder.image(frame_rgb)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    video.release()
    scan_video.release()
    cv.destroyAllWindows()


def blend_overlay(background, overlay):
    """Blend overlay video (with alpha if available) on background."""
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[:, :, c])
    else:
        background = cv.addWeighted(background, 0.6, overlay, 0.4, 0)
    return background


if st.button("Start Scanning"):
    faceScanner()

if st.button("Stop Scanning"):
    video.release()
    scan_video.release()
    cv.destroyAllWindows()
