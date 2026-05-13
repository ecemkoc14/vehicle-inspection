import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="SmartInspect Pro", layout="wide")
st.title("Next-Gen Inspection: Multi-Lane Queue Analytics")

# 1. Lane Setup & Live Occupancy
if 'lane_data' not in st.session_state:
    st.session_state.lane_data = {
        "Lane 1 (Mechanical)": {"wait": 15, "load": 40},
        "Lane 2 (Body & Paint)": {"wait": 10, "load": 25},
        "Lane 3 (EV & Electronic)": {"wait": 20, "load": 65}
    }

# Sidebar: Hourly Intensity Chart (Dynamic to current time)
current_hour = datetime.now().hour
hours = [(current_hour + i) % 24 for i in range(-4, 5)]
occupancy_values = [20, 35, 55, 85, 98, 75, 45, 25, 15]

st.sidebar.header("Station Real-Time Status")
st.sidebar.write(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
fig_time = px.line(x=hours, y=occupancy_values, labels={'x': 'Hour', 'y': 'Station Load %'})
fig_time.update_traces(line_color='#FF4B4B', mode='lines+markers')
st.sidebar.plotly_chart(fig_time, use_container_width=True)

# 2. Digital Diagnostic Entry
st.subheader("1. Vehicle Triage & Visual Analysis")
inspection_domain = st.selectbox("Assign Inspection Domain", ["Exterior (Body)", "Engine/Battery Compartment", "Interior/Electronics"])
uploaded_file = st.file_uploader("Capture or Upload Vehicle Data", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # AI Detection Core
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
            
            if area < 5000:
                color, label = (0, 255, 0), "Minor" # Green
                severity_counts["Minor"] += 1
            elif area < 20000:
                color, label = (255, 0, 0), "Moderate" # Blue
                severity_counts["Moderate"] += 1
            else:
                color, label = (0, 0, 255), "Critical" # Red
                severity_counts["Critical"] += 1
                
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 3)

    result_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
    c1, c2 = st.columns(2)
    with c1: st.image(img, caption="Input Data", use_container_width=True)
    with c2: st.image(result_img, caption="AI Structural Mapping", use_container_width=True)

    st.divider()

    # 3. Decision Support & Lane Assignment
    st.subheader("2. Lane Routing & Priority Verdict")
    
    if total_damage_area > 0:
        if total_damage_area < 8000:
            final_sev, final_col, wait_add = "MINOR", "green", 15
        elif total_damage_area < 30000:
            final_sev, final_col, wait_add = "MODERATE", "blue", 35
        else:
            final_sev, final_col, wait_add = "CRITICAL", "red", 60

        # Logic to pick the correct lane
        if inspection_domain == "Exterior (Body)":
            assigned_lane = "Lane 2 (Body & Paint)"
        elif inspection_domain == "Engine/Battery Compartment":
            assigned_lane = "Lane 1 (Mechanical)"
        else:
            assigned_lane = "Lane 3 (EV & Electronic)"

        current_lane_wait = st.session_state.lane_data[assigned_lane]["wait"]
        total_wait = current_lane_wait + wait_add

        # Dashboard Visuals
        col_metrics, col_chart = st.columns([1, 1])
        
        with col_metrics:
            st.markdown(f"### Target: :{final_col}[{assigned_lane}]")
            st.markdown(f"**Severity Level:** :{final_col}[{final_sev}]")
            st.metric("Total Estimated Wait", f"{total_wait} mins", delta=f"+{wait_add} diagnostic time")
            st.info(f"AI suggests {assigned_lane} due to {inspection_domain} detection.")

        with col_chart:
            df_pie = pd.DataFrame(list(severity_counts.items()), columns=['Severity', 'Count'])
            fig_pie = px.pie(df_pie, values='Count', names='Severity', 
                             color='Severity',
                             color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'},
                             hole=0.4)
            fig_pie.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig_pie, use_container_width=True)
            
    else:
        st.success("### No Anomalies: Redirecting to Lane 4 (Express)")
        st.metric("Total Estimated Wait", "8 mins")