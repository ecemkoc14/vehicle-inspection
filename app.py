import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="SmartInspect Analytics", layout="wide")
st.title("Advanced Inspection & Lane Management System")

# 1. Sidebar: Hourly Intensity Chart (Dynamic to current time)
current_hour = datetime.now().hour
hours = [(current_hour + i) % 24 for i in range(-4, 5)]
occupancy_data = [15, 25, 45, 85, 95, 75, 40, 20, 10] 

st.sidebar.header("Station Real-Time Status")
st.sidebar.write(f"System Sync: {datetime.now().strftime('%H:%M')}")
st.sidebar.subheader("Hourly Traffic Density")
fig_time = px.line(x=hours, y=occupancy_data, labels={'x': 'Hour', 'y': 'Station Load %'})
fig_time.update_traces(line_color='#FF4B4B', mode='lines+markers')
st.sidebar.plotly_chart(fig_time, use_container_width=True)

# 2. Main Diagnostic Section
st.subheader("1. Digital Triage & Visual Analysis")
inspection_domain = st.selectbox("Assign Inspection Domain", ["Exterior (Body)", "Engine/Battery Compartment", "Interior/Electronics"])
uploaded_file = st.file_uploader("Upload Inspection Data", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # High Precision Detection Logic
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 35, 110)
    kernel = np.ones((3,3), np.uint8)
    enhanced_edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(enhanced_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_damage_area = 0
    severity_counts = {"Minor": 0, "Moderate": 0, "Critical": 0}
    
    for c in contours:
        area = cv2.contourArea(c)
        if 400 < area < 60000:
            x, y, w, h = cv2.boundingRect(c)
            total_damage_area += area
            
            # Severity Logic & Bounding Boxes
            if area < 5000:
                color, label = (0, 255, 0), "Minor"
                severity_counts["Minor"] += 1
            elif area < 20000:
                color, label = (255, 0, 0), "Moderate"
                severity_counts["Moderate"] += 1
            else:
                color, label = (0, 0, 255), "Critical"
                severity_counts["Critical"] += 1
                
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 3)

    result_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
    # 3. Decision Support & Analytics Dashboard
    st.divider()
    st.subheader("2. Diagnostic Result & Queue Assignment")
    
    # Metrics and Pie Chart
    col_metrics, col_chart = st.columns([1, 1])
    
    if total_damage_area > 0:
        if total_damage_area < 8000:
            f_sev, f_col, wait_add = "MINOR", "green", 15
        elif total_damage_area < 30000:
            f_sev, f_col, wait_add = "MODERATE", "blue", 35
        else:
            f_sev, f_col, wait_add = "CRITICAL", "red", 60

        with col_metrics:
            st.markdown(f"### Verdict: :{f_col}[{f_sev} DETECTED]")
            target_lane = "Lane 2 (Body)" if inspection_domain == "Exterior (Body)" else "Lane 1 (Mechanical)"
            st.write(f"**Target Station:** {target_lane}")
            st.metric("Est. Additional Process Time", f"{wait_add} mins")
            st.info("The vehicle has been prioritized in the queue based on AI structural assessment.")

        with col_chart:
            df_pie = pd.DataFrame(list(severity_counts.items()), columns=['Severity', 'Count'])
            fig_pie = px.pie(df_pie, values='Count', names='Severity', 
                             color='Severity',
                             color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'},
                             hole=0.4, title="Anomaly Distribution")
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Analysis Output Display at the Bottom
        st.divider()
        st.subheader("3. Precise Visual Mapping")
        c_raw, c_proc = st.columns(2)
        c_raw.image(img, caption="Original Input", use_container_width=True)
        c_proc.image(result_img, caption="AI Detected Anomalies", use_container_width=True)
            
    else:
        st.success("### [NO ANOMALIES DETECTED]")
        st.write("Directing vehicle to Lane 4 (Express Exit).")
        st.metric("Total Estimated Wait", "8 mins")