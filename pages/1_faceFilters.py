import cv2 as cv
import numpy as np
import streamlit as st



st.title("Live Face Emoji Overlay")
face_detect = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")

video = cv.VideoCapture(0)

if "filter_index" not in st.session_state:
        st.session_state.filter_index = 1

filtersList = [
        'faces/filter0.png', 'faces/filter1.png', 'faces/filter2.png',
        'faces/filter3.png', 'faces/filter4.png', 'faces/filter5.png', 'faces/filter6.png'
    ]

def load_filter():
        return cv.imread(filtersList[st.session_state.filter_index], cv.IMREAD_UNCHANGED)


emoji = load_filter()


col1, col2 = st.columns([1, 5])
with col1:
        if st.button("⬅️"):
            st.session_state.filter_index = (st.session_state.filter_index - 1) % len(filtersList)
with col2:
        if st.button("➡️"):
            st.session_state.filter_index = (st.session_state.filter_index + 1) % len(filtersList)

frame_placeholder = st.empty()

st.markdown(
        """
        <style>
        .stApp {
            background-color: black;  
            color: white;                
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def gen_frames(): 
        while True:
            isTrue, frame = video.read()
            if not isTrue:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            emoji = load_filter()

            for (x, y, w, h) in faces:
                scale = 1.5
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv.resize(emoji, (new_w, new_h))

                x_offset = x - int((new_w - w) / 2)
                y_offset = y - int((new_h - h) / 2) - 30

                x_offset = max(0, x_offset)
                y_offset = max(0, y_offset)
                if x_offset + new_w > frame.shape[1]: new_w = frame.shape[1] - x_offset
                if y_offset + new_h > frame.shape[0]: new_h = frame.shape[0] - y_offset

                resized = cv.resize(resized, (new_w, new_h))
                b, g, r, a = cv.split(resized)
                overlay_color = cv.merge((b, g, r))
                alpha = a.astype(float) / 255.0
                alpha_inv = 1.0 - alpha

                roi = frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
                for c in range(3):
                    roi[:, :, c] = (alpha * overlay_color[:, :, c] +
                                    alpha_inv * roi[:, :, c])

                frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            flip = cv.flip(frame, 1)
            frame_placeholder.image(flip)


if st.button("Start"):
        gen_frames()
if st.button("End"):
        video.release()
        cv.destroyAllWindows()




