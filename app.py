import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import time

# --- SESSION STATE FOR SIMULATION ---
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = {
        'Lane_1': [], 'Lane_2': [], 'Lane_3': [],
        'Processed': 0, 'Savings': 0
    }

st.set_page_config(page_title="DeepCheck: AI Station Twin", layout="wide")

# --- HEADER SECTION ---
st.title("🏭 Next-Gen Vehicle Inspection: Digital Twin Dashboard")
st.markdown("### Operational Efficiency & AI-Powered Anomaly Detection")

# --- TOP ROW: DASHBOARD METRICS (PROFESSIONAL VIEW) ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Avg. Service Time", "18.5 min", "-2.1 min")
m2.metric("System Throughput", f"{st.session_state.sim_data['Processed']} Units", "+12%")
m3.metric("AI Accuracy Rate", "98.4%", "Stable")
m4.metric("Total Time Saved", f"{st.session_state.sim_data['Savings']} min", "Live")

# --- MIDDLE ROW: ANALYTICS GRAPHS ---
st.divider()
g1, g2 = st.columns(2)

with g1:
    # Hourly Traffic Simulation Data
    traffic_data = pd.DataFrame({
        'Hour': ['08:00', '10:00', '12:00', '14:00', '16:00', '18:00'],
        'Vehicles': [12, 45, 30, 55, 40, 20]
    })
    fig1 = px.line(traffic_data, x='Hour', y='Vehicles', title="Hourly Vehicle Traffic (Station Peak Analysis)", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

with g2:
    # Lane Distribution Data
    lane_data = pd.DataFrame({
        'Station Lane': ['Lane 1 (Mech)', 'Lane 2 (Body)', 'Lane 3 (EV)'],
        'Queue Length': [len(st.session_state.sim_data['Lane_1']), len(st.session_state.sim_data['Lane_2']), len(st.session_state.sim_data['Lane_3'])]
    })
    fig2 = px.bar(lane_data, x='Station Lane', y='Queue Length', color='Station Lane', title="Live Queue Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# --- BOTTOM SECTION: LIVE INSPECTION & SIMULATION ---
st.divider()
st.subheader("🔍 Live AI Diagnostic & Queue Simulator")

c1, c2 = st.columns([1, 1])

with c1:
    insp_domain = st.selectbox("Inspection Focus", ["Exterior (Body)", "Engine/Battery Compartment", "Interior"])
    uploaded_file = st.file_uploader("Upload Inspection Data (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if st.button("Clear Station Data"):
        st.session_state.sim_data = {'Lane_1': [], 'Lane_2': [], 'Lane_3': [], 'Processed': 0, 'Savings': 0}
        st.rerun()

with c2:
    if uploaded_file:
        img = Image.open(uploaded_file)
        frame = np.array(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Damage Detection Algorithm
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        damage_sum = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 500)
        
        # Severity Logic
        if damage_sum > 20000:
            sev, color, p_time = "CRITICAL", "red", 45
        elif damage_sum > 5000:
            sev, color, p_time = "MODERATE", "orange", 25
        else:
            sev, color, p_time = "MINOR", "green", 12

        st.image(img, caption=f"AI Diagnostic: {sev} Severity detected", use_container_width=True)
        
        if st.button("Approve & Route Vehicle"):
            target = 'Lane_2' if insp_domain == "Exterior (Body)" else 'Lane_1'
            st.session_state.sim_data[target].append(p_time)
            st.session_state.sim_data['Processed'] += 1
            st.session_state.sim_data['Savings'] += 15
            st.toast(f"Vehicle routed to {target}!", icon='✅')
            time.sleep(0.5)
            st.rerun()

# --- FOOTER METADATA ---
st.sidebar.markdown("### System Metadata")
st.sidebar.write("**Algorithm:** Hybrid Vision-IE Optimization")
st.sidebar.write("**Module:** Smart City/Digital Twin")
st.sidebar.write("**Status:** Active")