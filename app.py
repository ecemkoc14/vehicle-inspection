import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Smart Gate Inspection", layout="wide")

# --- HEADER SECTION ---
st.title("Smart Gate: Automated Vehicle Inspection & Queue Management")

# --- STEP 1: VEHICLE CLASSIFICATION ---
st.subheader("1. Vehicle & Gate Selection")
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    vehicle_type = st.selectbox("Vehicle Classification", ["Passenger Car", "Electric Vehicle (EV)", "Heavy Truck / Logistics"])
with col_info2:
    gate_selection = st.selectbox("Assign to Gate", ["Gate A (Fast Track)", "Gate B (Standard)", "Gate C (Technical)"])
with col_info3:
    st.metric("Daily Throughput", "142 Vehicles", "+12%")

# --- GATE STATUS INDICATORS ---
st.write("### Live Gate Status")
gate_col1, gate_col2, gate_col3, gate_col4 = st.columns(4)
gate_col1.success("Gate 1: OPEN")
gate_col2.success("Gate 2: OPEN")
gate_col3.warning("Gate 3: BUSY (8m)")
gate_col4.error("Gate 4: CLOSED")

st.divider()

# --- STEP 2: IMAGE ANALYSIS ---
st.subheader("2. Visual Structural Analysis")

# Sidebar: Intensity Analytics
current_hour = datetime.now().hour
hours = [(current_hour + i) % 24 for i in range(-4, 5)]
occupancy = [20, 35, 60, 90, 85, 70, 45, 25, 10]
st.sidebar.header("Station Load Analytics")
fig_side = px.line(x=hours, y=occupancy, labels={'x': 'Hour', 'y': 'Load %'}, title="Hourly Intensity")
fig_side.update_traces(line_color='red')
st.sidebar.plotly_chart(fig_side, use_container_width=True)

uploaded_file = st.file_uploader("Upload Inspection Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # High-Precision Algorithm
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 110)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = 0
    severity_map = {"Minor": 0, "Moderate": 0, "Critical": 0}
    
    for c in contours:
        area = cv2.contourArea(c)
        if 500 < area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            total_area += area
            if area < 5000:
                color, label = (0, 255, 0), "Minor"
                severity_map["Minor"] += 1
            elif area < 20000:
                color, label = (255, 0, 0), "Moderate"
                severity_map["Moderate"] += 1
            else:
                color, label = (0, 0, 255), "Critical"
                severity_map["Critical"] += 1
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 3)

    processed_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # --- RESULTS & PIE CHART ---
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.image(processed_img, caption="AI Detection Mapping", use_container_width=True)
        
    with res_col2:
        st.write("### Anomaly Distribution")
        fig_pie = px.pie(values=list(severity_map.values()), 
                         names=list(severity_map.keys()),
                         color=list(severity_map.keys()),
                         color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # --- STEP 3: QUEUE VERDICT ---
    st.subheader("3. Final Queue Verdict")
    if total_area > 0:
        if total_area < 8000:
            sev_level, wait = "MINOR", 12
        elif total_area < 25000:
            sev_level, wait = "MODERATE", 25
        else:
            sev_level, wait = "CRITICAL", 50
            
        st.error(f"**ISSUE DETECTED:** {sev_level} structural anomaly found on {vehicle_type}.")
        st.info(f"**SYSTEM ACTION:** Vehicle routed to {gate_selection}. Estimated processing time: **{wait} minutes**.")
    else:
        st.success(f"**CLEAN SCAN:** No issues found for {vehicle_type}. Proceed through Gate 1.")
        st.metric("Wait Time", "4 mins")