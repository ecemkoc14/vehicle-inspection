import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="SmartInspect Pro", layout="wide")
st.title("Automated Inspection & Analytics Dashboard")

# 1. Dynamic Queue Data (Centered around current time)
current_hour = datetime.now().hour
hours = [(current_hour + i) % 24 for i in range(-4, 5)]
occupancy_data = [15, 22, 45, 80, 95, 70, 40, 20, 10] # Simulated intensity peak at current time

# Sidebar: Live Station Metrics
st.sidebar.header("Station Real-Time Status")
st.sidebar.write(f"Current Time: {datetime.now().strftime('%H:%M')}")
st.sidebar.subheader("Hourly Intensity")
fig_time = px.line(x=hours, y=occupancy_data, labels={'x': 'Hour', 'y': 'Occupancy %'})
fig_time.update_traces(line_color='#FF4B4B')
st.sidebar.plotly_chart(fig_time, use_container_width=True)

# Main App Logic
st.subheader("1. Digital Diagnostic Entry")
inspection_type = st.selectbox("Inspection Domain", ["Exterior (Body)", "Engine/Battery Compartment", "Interior/Electronics"])
uploaded_file = st.file_uploader("Upload Inspection Data", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = np.ones((3,3), np.uint8)
    enhanced_edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(enhanced_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_damage_area = 0
    severity_counts = {"Minor": 0, "Moderate": 0, "Critical": 0}
    
    for c in contours:
        area = cv2.contourArea(c)
        if 400 < area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            total_damage_area += area
            
            # Color Logic (BGR Format for OpenCV)
            if area < 5000:
                color = (0, 255, 0) # Green (Minor)
                severity_counts["Minor"] += 1
            elif area < 20000:
                color = (255, 0, 0) # Blue (Moderate)
                severity_counts["Moderate"] += 1
            else:
                color = (0, 0, 255) # Red (Critical)
                severity_counts["Critical"] += 1
                
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 3)

    result_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Live Field Input", use_container_width=True)
    with col2:
        st.image(result_img, caption="AI Structural Diagnostics", use_container_width=True)

    st.divider()

    # Analytics Section
    st.subheader("2. Decision Support & Severity Analytics")
    
    if total_damage_area > 0:
        # Determine overall severity
        if total_damage_area < 8000:
            final_sev, final_col, wait_add = "MINOR", "green", 15
        elif total_damage_area < 30000:
            final_sev, final_col, wait_add = "MODERATE", "blue", 35
        else:
            final_sev, final_col, wait_add = "CRITICAL", "red", 60

        # Distribution Pie Chart
        df_pie = pd.DataFrame(list(severity_counts.items()), columns=['Severity', 'Count'])
        fig_pie = px.pie(df_pie, values='Count', names='Severity', 
                         color='Severity',
                         color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'},
                         title="Detection Distribution")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.markdown(f"### Verdict: :{final_col}[{final_sev} STATUS]")
            st.write(f"**Target Station:** {'Lane 2 (Body)' if inspection_type == 'Exterior (Body)' else 'Lane 1 (Mechanical)'}")
            st.metric("Estimated Process Time", f"{wait_add} mins")
            st.info("The vehicle has been prioritized based on structural integrity analysis.")
    else:
        st.success("### Verdict: [NO ANOMALIES DETECTED]")
        st.write("Directing to Express Lane.")