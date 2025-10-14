import PIL.Image
import google.generativeai as genai
import cv2 as cv 
import streamlit as st

st.title("Webcam Live Feed")
st.text("Click on the button to capture an image from your webcam")
placeholder = st.empty()
if st.button('Capture Image'):
    video = cv.VideoCapture(0)
    ret, frame = video.read()
    flip = cv.flip(frame, 1)
    cv.imwrite('frame.jpg', flip)
    placeholder.image(flip)
    video.release()
    genai.configure(api_key="AIzaSyCkeTgoTp5llhS98bc380Za6BXKulO829I")
    img=PIL.Image.open('frame.jpg')
    model = genai.GenerativeModel('gemini-flash-latest')
    response=model.generate_content(img)
    st.text(response.text)