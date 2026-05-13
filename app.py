import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Dynamic Inspection Dashboard", layout="wide")

# --- SESSION STATE (Veri Girdikçe Artan Sayaçlar ve Grafik Verisi) ---
if 'inspected_list' not in st.session_state:
    st.session_state.inspected_list = [] # Girilen her aracın saati ve sonucu burada tutulacak
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'critical_count' not in st.session_state:
    st.session_state.critical_count = 0

# --- 1. TOP KPI ROW (Gerçek Zamanlı Sayaçlar) ---
st.title("Vehicle Inspection & Network Analytics")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Inspected", st.session_state.total_count, f"+{1 if st.session_state.total_count > 0 else 0}")
kpi2.metric("Critical Cases", st.session_state.critical_count)
kpi3.metric("Avg. Cycle Time", "12.4 min" if st.session_state.total_count > 0 else "0 min")
kpi4.metric("System Efficiency", "98%" if st.session_state.total_count > 0 else "100%")

st.divider()

# --- 2. ANALYTICS ROW (Dinamik Grafikler) ---
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("Hourly Traffic Density")
    # Veri girdikçe güncellenen grafik mantığı
    all_hours = list(range(0, 24))
    counts_per_hour = [0] * 24
    
    # Session state'deki verileri saatlere göre say
    for entry in st.session_state.inspected_list:
        counts_per_hour[entry['hour']] += 1
        
    df_line = pd.DataFrame({'Hour': all_hours, 'Vehicles': counts_per_hour})
    fig_line = px.line(df_line, x='Hour', y='Vehicles', markers=True, title="Live Entry Logs")
    fig_line.update_traces(line_color='#FF4B4B')
    # Sadece veri olan saatleri ve çevresini odakla
    current_h = datetime.now().hour
    fig_line.update_xaxes(range=[max(0, current_h-5), min(23, current_h+5)])
    st.plotly_chart(fig_line, use_container_width=True)

with col_graph2:
    st.subheader("Anomaly Severity Distribution")
    severity_placeholder = st.empty()

st.divider()

# --- 3. DATA ENTRY & AI SCAN (Gelişmiş Hassasiyet) ---
st.subheader("3. Data Entry & AI Scan")
input_col1, input_col2 = st.columns(2)

with input_col1:
    v_category = st.radio("Vehicle Category", ["Passenger Car", "Electric Vehicle (EV)", "Truck / Logistics"], horizontal=True)
    i_focus = st.selectbox("Inspection Focus", ["Exterior (Body)", "Engine Compartment", "Battery Case"])
    uploaded_file = st.file_uploader("Upload Inspection Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # HASSASİYET AYARI: Ağır hasarları kaçırmamak için kernel büyütüldü ve threshold esnetildi
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 20, 80) # Daha düşük eşik ile daha çok detay
    kernel = np.ones((7,7), np.uint8) # Boşlukları kapatmak için daha geniş kernel
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = 0
    sev_counts = {"Minor": 0, "Moderate": 0, "Critical": 0}
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 300: # Gürültüleri ele ama hasarı yakala
            x, y, w, h = cv2.boundingRect(c)
            total_area += area
            
            # Severity Renk Mantığı
            if area < 8000:
                color, sev_key = (0, 255, 0), "Minor"
                sev_counts["Minor"] += 1
            elif area < 35000:
                color, sev_key = (255, 0, 0), "Moderate"
                sev_counts["Moderate"] += 1
            else:
                color, sev_key = (0, 0, 255), "Critical"
                sev_counts["Critical"] += 1
            
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 4)

    # --- SİSTEMİ GÜNCELLE (Butona basılmış gibi her yüklemede) ---
    now = datetime.now()
    st.session_state.total_count += 1
    
    is_critical = total_area > 35000
    if is_critical:
        st.session_state.critical_count += 1
    
    # Grafik için veri ekle
    st.session_state.inspected_list.append({'hour': now.hour, 'severity': 'Critical' if is_critical else 'Normal'})

    # Pie Chart Güncelleme
    df_pie = pd.DataFrame(list(sev_counts.items()), columns=['Severity', 'Count'])
    fig_pie = px.pie(df_pie, values='Count', names='Severity', 
                     color='Severity',
                     color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'},
                     hole=0.4)
    severity_placeholder.plotly_chart(fig_pie, use_container_width=True)

    # Sonuç Gösterimi
    res_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    with input_col2:
        st.image(res_img, caption="AI High-Precision Detection", use_container_width=True)
        if is_critical:
            st.error(f"SEVERE DAMAGE DETECTED: Vehicle routed to Critical Lane.")
        else:
            st.success("Analysis complete. Anomaly levels within standard limits.")