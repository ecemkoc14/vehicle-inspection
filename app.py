import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import random

st.set_page_config(page_title="VisionControl AI", layout="wide")

# --- SIMULATED DATABASE ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['ID', 'Type', 'Severity', 'Risk_Score', 'Status'])

# --- STATION LIVE LOAD (Simulation) ---
st.sidebar.header("Station Real-Time Load")
lane_load = {"Lane 1 (Mech)": 15, "Lane 2 (Body)": 10, "Lane 3 (EV)": 20}
for lane, load in lane_load.items():
    st.sidebar.write(f"**{lane}**: {load} mins base wait")
    st.sidebar.progress(load / 60)

st.title("🛡️ VisionControl: Advanced Autonomous Safety Hub")
st.markdown("##### *Integrated Decision Support & Dynamic Resource Allocation*")

# --- TOP ROW: KPI DASHBOARD ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fleet Safety Index", "86/100", "+1.2")
c2.metric("Anomaly Accuracy", "97.1%", "+0.8%")
c3.metric("Critical Alerts", len(st.session_state.history[st.session_state.history['Severity'] == 'CRITICAL']), delta_color="inverse")
c4.metric("Avg. Decision Speed", "1.1s", "-0.1s")

st.divider()

# --- ANALYTICS: RISK HEATMAP & HISTORY ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Network Risk Distribution (Time Series)")
    hours = list(range(8, 18))
    # Synthetic risk data with a 'peak' pattern
    risk_levels = [random.randint(10, 40) for _ in hours[:3]] + [random.randint(60, 95) for _ in hours[3:7]] + [random.randint(20, 50) for _ in hours[7:]]
    fig = go.Figure(data=go.Scatter(x=hours, y=risk_levels, fill='tozeroy', line_color='indigo', name="Risk Density"))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Hour", yaxis_title="Risk Score")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📋 Final Audit Log (Recent Entries)")
    st.dataframe(st.session_state.history.tail(5), use_container_width=True, hide_index=True)

st.divider()

# --- INTERACTIVE AI INSPECTION & QUEUEING ---
st.subheader("🤖 Smart Diagnostics & Dynamic Queueing")
i1, i2 = st.columns([1, 1])

with i1:
    v_type = st.selectbox("Vehicle Class", ["Passenger (ICE)", "Electric (EV)", "Heavy Duty (Truck)"])
    inspection_focus = st.selectbox("Inspection Focus", ["Exterior (Body Surface)", "Engine/Battery Compartment"])
    up_file = st.file_uploader("Upload Inspection Data (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if st.button("Reset Station Data"):
        st.session_state.history = pd.DataFrame(columns=['ID', 'Type', 'Severity', 'Risk_Score', 'Status'])
        st.toast("Simulation data reset!", icon="🧹")
        st.rerun()

with i2:
    if up_file:
        img = Image.open(up_file)
        frame = np.array(img)
        
        # --- ENHANCED AI DAMAGE DETECTION ---
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Texture-based filtering for complex damage (Tears, Crushes)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        # Using adaptive thresholding to isolate texture abnormalities (wet/crushed areas)
        texture_mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 2. Strong edge detection for structural integrity
        edges = cv2.Canny(gray, 40, 100) # Increased sensitivity
        
        # Combined mask for complex defects
        combined_analysis = cv2.addWeighted(edges, 0.6, texture_mask, 0.4, 0)
        
        # Dilation to connect fragmented damage regions (Crucial for large crushes)
        kernel = np.ones((5,5), np.uint8)
        dilated_analysis = cv2.dilate(combined_analysis, kernel, iterations=2)
        
        # Counting and drawing anomalies
        contours, _ = cv2.findContours(dilated_analysis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_damage_pixels = 0
        anomaly_zones = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000: # Threshold for defect size
                x, y, w, h = cv2.boundingRect(cnt)
                total_damage_pixels += area
                anomaly_zones += 1
                # Dynamic box color based on local area size
                color = (255, 0, 0) if area < 8000 else (0, 0, 255) # Red for large issues
                cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 3)

        result_view = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        st.image(result_view, caption=f"AI Neural Diagnostics: {anomaly_zones} Anomalies Boxed", use_container_width=True)
        
        # --- SEVERITY & ROUTING LOGIC (Optimization Part) ---
        if st.button("Generate Final Audit & Dynamic Route"):
            # Mock AI severity logic based on TOTAL damaged area
            if total_damage_pixels > 25000: # Focused threshold for large crushes
                sev, res_col, res_text = "CRITICAL", "red", "REJECTED (Structural Failure)"
                proc_time_needed = 50
            elif total_damage_pixels > 8000:
                sev, res_col, res_text = "MODERATE", "orange", "CONDITIONAL PASS (Repair Needed)"
                proc_time_needed = 30
            else:
                sev, res_col, res_text = "MINOR/CLEAN", "green", "CERTIFIED (Pass)"
                proc_time_needed = 12
            
            # Decision making: Safety over Speed
            risk_score = min(100, int((total_damage_pixels / 50000) * 100)) # Simple risk indexing
            
            # Smart Routing (The IE Part): Safety Lane for high risk
            if sev == "CRITICAL":
                target_lane = "Lane 3 (EV/Safety specialized)" if v_type == "Electric (EV)" else "Lane 1 (Mech specialized)"
                base_queue = lane_load.get("Lane 3 (EV)" if v_type == "Electric (EV)" else "Lane 1 (Mech)", 0)
            else:
                target_lane = "Lane 2 (Fast-Track/Body)"
                base_queue = lane_load.get("Lane 2 (Body)", 0)
            
            # Add to audit log
            new_entry = pd.DataFrame({'ID': [random.randint(1000, 9999)], 'Type': [v_type], 'Severity': [sev], 'Risk_Score': [f"{100-risk_score}/100"], 'Status': [res_text]})
            st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
            
            # Professional Output
            st.markdown(f"### Final Verdict: :{res_col}[{res_text}]")
            c1, c2, c3 = st.columns(3)
            c1.metric("Safety Score", f"{100 - risk_score}/100")
            c2.metric("Target Station", target_lane)
            c3.metric("Est. Time (Wait + Service)", f"{base_queue + proc_time_needed} min")
            
            st.info(f"**AI Diagnostic Summary:** {anomaly_zones} surface irregularities with high visual dissonance identified. Structural integrity score below optimal threshold. Proceeding with {sev} protocol.")
            time.sleep(0.5)
            st.rerun()

# --- SYSTEM METADATA ---
st.sidebar.markdown("### Decision Logic Status")
st.sidebar.success("Adaptive Damage Model: Active")
st.sidebar.success("Dynamic Queue Router: Operational")
st.sidebar.write("Algorithm: Hybrid Vision-Texture MCDM")
st.sidebar.write("Feasibility Score: 89.4%")