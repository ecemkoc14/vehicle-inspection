import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Inspection Dashboard", layout="wide")

# --- 1. TOP KPI ROW (En Üstteki İstatistik Kapıları) ---
st.title("Vehicle Inspection & Network Analytics")

# Sayıların her yüklemede artması için session_state kullanıyoruz
if 'total_inspected' not in st.session_state:
    st.session_state.total_inspected = 142
if 'critical_cases' not in st.session_state:
    st.session_state.critical_cases = 12

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Inspected", st.session_state.total_inspected, "+1")
kpi2.metric("Critical Cases", st.session_state.critical_cases, "High Priority")
kpi3.metric("Avg. Cycle Time", "18.5 min", "-2.1m")
kpi4.metric("System Efficiency", "94%", "+1.2%")

st.divider()

# --- 2. ANALYTICS ROW (Orta Bölüm: Grafikler Yan Yana) ---
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("Hourly Traffic Density")
    # SAAT DÜZELTMESİ: Şu anki saati merkez alır (11:00 - 12:00 arası için dinamik)
    current_hour = datetime.now().hour
    hours = [(current_hour + i) % 24 for i in range(-4, 5)]
    # Yoğunluk grafiği şu anki saatinde (merkezde) zirve yapar
    occupancy = [20, 35, 55, 85, 98, 75, 45, 25, 15] 
    
    fig_line = px.line(x=hours, y=occupancy, labels={'x': 'Hour (Live Sync)', 'y': 'Station Load %'})
    fig_line.update_traces(line_color='#FF4B4B', mode='lines+markers')
    st.plotly_chart(fig_line, use_container_width=True)

with col_graph2:
    st.subheader("Anomaly Severity Distribution")
    # Pie Chart için placeholder (aşağıdaki analizden sonra dolacak)
    severity_placeholder = st.empty()

st.divider()

# --- 3. INPUT & ANALYSIS SECTION (En Alt Bölüm) ---
st.subheader("3. Data Entry & AI Scan")
input_col1, input_col2 = st.columns(2)

with input_col1:
    # Araç ve Bölge Seçimi
    v_category = st.radio("Vehicle Category", ["Passenger Car", "Electric Vehicle (EV)", "Truck / Logistics"], horizontal=True)
    i_focus = st.selectbox("Inspection Focus", ["Exterior (Body)", "Engine Compartment", "Underbody / Battery Case"])
    uploaded_file = st.file_uploader("Upload Inspection Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # Hassas Analiz Algoritması
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 40, 110)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = 0
    sev_counts = {"Minor": 0, "Moderate": 0, "Critical": 0}
    
    for c in contours:
        area = cv2.contourArea(c)
        if 400 < area < 60000:
            x, y, w, h = cv2.boundingRect(c)
            total_area += area
            
            # RENK DÜZELTMESİ (OpenCV BGR formatı)
            if area < 5000:
                color, label = (0, 255, 0), "Minor" # Yeşil
                sev_counts["Minor"] += 1
            elif area < 20000:
                color, label = (255, 0, 0), "Moderate" # Mavi (BGR'de 255,0,0 mavidir)
                sev_counts["Moderate"] += 1
            else:
                color, label = (0, 0, 255), "Critical" # Kırmızı
                sev_counts["Critical"] += 1
                
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 3)

    # Verileri Güncelle
    st.session_state.total_inspected += 1
    
    # PIE CHART RENK DÜZELTMESİ
    df_pie = pd.DataFrame(list(sev_counts.items()), columns=['Severity', 'Count'])
    fig_pie = px.pie(df_pie, values='Count', names='Severity', 
                     color='Severity',
                     color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'},
                     hole=0.4)
    severity_placeholder.plotly_chart(fig_pie, use_container_width=True)

    # Analiz Görüntüsünü İşle ve Göster
    res_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
    with input_col2:
        st.image(res_img, caption="AI Structural Diagnostics", use_container_width=True)
        
        if total_area > 20000:
            st.error(f"CRITICAL DAMAGE DETECTED on {v_category}")
            st.session_state.critical_cases += 1
        elif total_area > 0:
            st.warning(f"Moderate/Minor anomalies detected. Reviewing {i_focus}...")
        else:
            st.success(f"No significant issues found for this {v_category}.")