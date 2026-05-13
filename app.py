import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Smart Inspection Dashboard", layout="wide")

# --- SESSION STATE (Dinamik Veri Yönetimi) ---
if 'inspected_list' not in st.session_state:
    st.session_state.inspected_list = []
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'critical_count' not in st.session_state:
    st.session_state.critical_count = 0

# --- HEADER & KPI ROW ---
st.title("Vehicle Inspection & Network Analytics")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Inspected", st.session_state.total_count)
kpi2.metric("Critical Cases", st.session_state.critical_count)
kpi3.metric("Avg. Cycle Time", "12.4 min" if st.session_state.total_count > 0 else "0 min")
kpi4.metric("System Efficiency", "98%" if st.session_state.total_count > 0 else "100%")

st.divider()

# --- ANALYTICS ROW ---
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("Hourly Traffic Density")
    all_hours = list(range(0, 24))
    counts_per_hour = [0] * 24
    for entry in st.session_state.inspected_list:
        counts_per_hour[entry['hour']] += 1
        
    df_line = pd.DataFrame({'Hour': all_hours, 'Vehicles': counts_per_hour})
    fig_line = px.line(df_line, x='Hour', y='Vehicles', markers=True)
    fig_line.update_traces(line_color='#FF4B4B')
    current_h = datetime.now().hour
    fig_line.update_xaxes(range=[max(0, current_h-4), min(23, current_h+4)])
    st.plotly_chart(fig_line, use_container_width=True)

with col_graph2:
    st.subheader("Anomaly Severity Distribution")
    severity_placeholder = st.empty()

st.divider()

# --- DATA ENTRY & CONTROLS ---
st.subheader("3. Data Entry & AI Scan")
input_col1, input_col2 = st.columns(2)

with input_col1:
    v_category = st.radio("Vehicle Category", ["Passenger Car", "Electric Vehicle (EV)", "Truck / Logistics"], horizontal=True)
    i_focus = st.selectbox("Inspection Focus", ["Exterior (Body)", "Engine Compartment", "Battery Case"])
    uploaded_file = st.file_uploader("Upload Inspection Image", type=["jpg", "jpeg", "png"])
    
    # --- BUTONLAR ---
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        update_btn = st.button("🔄 Update System & Log", use_container_width=True)
    with btn_col2:
        if st.button("🗑️ Reset Dashboard", use_container_width=True):
            st.session_state.inspected_list = []
            st.session_state.total_count = 0
            st.session_state.critical_count = 0
            st.rerun()

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # --- HASSASİYET DENGESİ ---
    # Kaput içi (Engine) için daha sakin, Dış (Exterior) için daha geniş tarama
    blur_k = (15, 15) if i_focus == "Engine Compartment" else (9, 9)
    blur = cv2.GaussianBlur(gray, blur_k, 0)
    
    # Kaput içinde gürültüyü azaltmak için threshold yükseltildi
    lower_thr = 50 if i_focus == "Engine Compartment" else 25
    edges = cv2.Canny(blur, lower_thr, 150)
    
    kernel_size = 3 if i_focus == "Engine Compartment" else 7
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = 0
    sev_counts = {"Minor": 0, "Moderate": 0, "Critical": 0}
    
    # Dinamik Kritik Eşik (Kaput içi daha zordur, eşiği yükseltiyoruz)
    critical_threshold = 50000 if i_focus == "Engine Compartment" else 30000

    for c in contours:
        area = cv2.contourArea(c)
        if area > 600:
            x, y, w, h = cv2.boundingRect(c)
            total_area += area
            
            if area < 10000:
                color, label = (0, 255, 0), "Minor"
                sev_counts["Minor"] += 1
            elif area < critical_threshold:
                color, label = (255, 0, 0), "Moderate"
                sev_counts["Moderate"] += 1
            else:
                color, label = (0, 0, 255), "Critical"
                sev_counts["Critical"] += 1
            
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 4)

    # Pie Chart Güncelleme (Her zaman göster)
    df_pie = pd.DataFrame(list(sev_counts.items()), columns=['Severity', 'Count'])
    fig_pie = px.pie(df_pie, values='Count', names='Severity', 
                     color='Severity',
                     color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'},
                     hole=0.4)
    severity_placeholder.plotly_chart(fig_pie, use_container_width=True)

    # Analiz Görüntüsü
    res_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    with input_col2:
        st.image(res_img, caption="AI Real-Time Diagnostics", use_container_width=True)

    # --- UPDATE LOGIC ---
    if update_btn:
        st.session_state.total_count += 1
        if total_area > critical_threshold:
            st.session_state.critical_count += 1
        st.session_state.inspected_list.append({'hour': datetime.now().hour})
        st.toast("System updated successfully!", icon="✅")
        st.rerun()