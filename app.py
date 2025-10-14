import streamlit as st

st.set_page_config(page_title="Face Filter App", page_icon="😎", layout="wide")

st.title("🎭 Welcome to Rohit's OpenCV Filters App!")
st.write("""
Select a page from the sidebar to start:
- **Face Filters Page** → Add live emoji overlays  
- **Eye Scanner Page** → Detect and highlight eyes
- **Motion Detector Page** → Detect motion in real-time 
""")