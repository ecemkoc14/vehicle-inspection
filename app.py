import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="SmartInspect Pro", layout="wide")
st.title("Next-Gen Inspection: Visual Analysis & Smart Queue Management")

# Sidebar for System Status
st.sidebar.header("Station Live Status")
st.sidebar.info("Lane 1 (Mechanical): 12 min wait")
st.sidebar.info("Lane 2 (Body & Paint): 5 min wait")
st.sidebar.info("Lane 3 (EV & Electronic): 8 min wait")

# Step 1: Image Upload
st.subheader("1. Digital Inspection Entry")
inspection_type = st.selectbox("Select Inspection Area", ["Exterior (Body)", "Engine/Battery Compartment", "Interior/Electronics"])
uploaded_file = st.file_uploader("Upload Inspection Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    
    # AI Analysis Processing
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # Enhanced Sensitivity Settings
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100) 
    
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anomaly_score = 0
    for c in contours:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            anomaly_score += 1

    result_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Input Data", use_container_width=True)
    with col2:
        st.image(result_img, caption="AI Feature Detection", use_container_width=True)

    st.divider()

    # Step 2: Smart Queue Routing
    st.subheader("2. Automated Routing & Queue Verdict")
    
    if anomaly_score > 0:
        if inspection_type == "Exterior (Body)":
            target_lane = "Lane 2 (Body & Chassis)"
            reason = f"Structural deformation detected ({anomaly_score} zones)."
        elif inspection_type == "Engine/Battery Compartment":
            target_lane = "Lane 1 (Mechanical)"
            reason = "Component irregularity detected."
        else:
            target_lane = "Lane 3 (EV & Electronic)"
            reason = "Electronic/Interior fault signature identified."

        st.warning(f"**STATUS:** Anomalies Detected.")
        st.error(f"**ROUTING:** Vehicle directed to **{target_lane}**")
        st.write(f"**Reason:** {reason}")
        st.metric(label="Estimated Inspection Time", value=f"{random.randint(20, 40)} mins")
    else:
        st.success("**STATUS:** No visible anomalies. Proceed to Fast-Track Lane.")
        st.write("**ROUTING:** Vehicle directed to **Lane 4 (Express Exit)**")
        st.metric(label="Estimated Inspection Time", value="8 mins")

    st.sidebar.success("Routing Algorithm: Active")