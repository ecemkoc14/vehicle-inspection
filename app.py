import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="SmartInspect AI", layout="wide")
st.title("Next-Gen Inspection: Severity-Based Analysis & Smart Queueing")

# Lane Status Tracking
st.sidebar.header("Station Real-Time Load")
lane_load = {"Lane 1": 15, "Lane 2": 10, "Lane 3": 20}
for lane, load in lane_load.items():
    st.sidebar.progress(load / 60)
    st.sidebar.write(f"{lane}: {load} mins base wait")

st.subheader("1. Digital Diagnostic Entry")
inspection_type = st.selectbox("Inspection Domain", ["Exterior (Body)", "Engine/Battery Compartment", "Interior/Electronics"])
uploaded_file = st.file_uploader("Upload Inspection Data", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # Advanced Filtering for Specific Defects (Tears & Leaks)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 40, 120)
    
    # Feature Enhancement
    kernel = np.ones((3,3), np.uint8)
    enhanced_edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(enhanced_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_damage_area = 0
    anomalies = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if 400 < area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            total_damage_area += area
            anomalies.append((x, y, w, h))
            # Dynamic Box Color based on local area size
            color = (255, 0, 0) if area < 5000 else (0, 0, 255)
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 2)

    result_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Live Feed", use_container_width=True)
    with col2:
        st.image(result_img, caption="AI Structural Analysis", use_container_width=True)

    st.divider()

    # Step 2: Severity Assessment & Queue Logic (IE Optimization)
    st.subheader("2. Diagnostic Result & Queue Optimization")
    
    # Calculating Severity based on total pixels affected
    if total_damage_area > 0:
        if total_damage_area < 5000:
            severity = "MINOR"
            base_time = 15
            status_color = "blue"
        elif total_damage_area < 25000:
            severity = "MODERATE"
            base_time = 30
            status_color = "orange"
        else:
            severity = "CRITICAL"
            base_time = 55
            status_color = "red"

        # Final Decision Logic
        target_lane = "Lane 2" if inspection_type == "Exterior (Body)" else "Lane 1"
        total_wait = base_time + lane_load.get(target_lane, 0)

        st.markdown(f"### Verdict: :{status_color}[{severity} ANOMALY DETECTED]")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Damage Index", f"{int(total_damage_area)}")
        c2.metric("Target Station", target_lane)
        c3.metric("Est. Process Time", f"{total_wait} min")
        
        st.info(f"**Analysis:** {inspection_type} shows structural non-compliance. Priority level adjusted for {severity} status.")
    else:
        st.success("### Verdict: [NO ANOMALIES DETECTED]")
        st.metric("Est. Process Time", "10 min")
        st.write("Vehicle directed to Fast-Track Lane (Lane 4).")