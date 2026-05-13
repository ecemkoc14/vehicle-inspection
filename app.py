import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Smart Inspection Dashboard", layout="wide")

# --- SESSION STATE (Veri Deposu) ---
if 'history' not in st.session_state:
    st.session_state.history = [] # Tüm geçmiş sonuçlar burada: {'hour': 12, 'sev': 'Critical'}
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'critical_count' not in st.session_state:
    st.session_state.critical_count = 0

# --- HEADER & KPI ROW ---
st.title("Vehicle Inspection & Network Analytics")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Inspected", st.session_state.total_count)
kpi2.metric("Critical Cases", st.session_state.critical_count)
kpi3.metric("Avg. Cycle Time", "12.4 min" if st.session_state.total_count > 0 else "0")
kpi4.metric("System Efficiency", "98%" if st.session_state.total_count > 0 else "100%")

st.divider()

# --- ANALYTICS ROW ---
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("Hourly Traffic Density")
    all_hours = list(range(0, 24))
    counts = [0] * 24
    for item in st.session_state.history:
        counts[item['hour']] += 1
    
    fig_line = px.line(x=all_hours, y=counts, markers=True, labels={'x':'Hour', 'y':'Vehicles'})
    fig_line.update_traces(line_color='#FF4B4B')
    # Mevcut saate odakla
    cur_h = datetime.now().hour
    fig_line.update_xaxes(range=[max(0, cur_h-4), min(23, cur_h+4)])
    st.plotly_chart(fig_line, use_container_width=True)

with col_graph2:
    st.subheader("Cumulative Severity Distribution")
    # Geçmiş verilerden pie chart oluştur
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        pie_data = df_hist['sev'].value_counts().reset_index()
        pie_data.columns = ['Severity', 'Count']
        fig_pie = px.pie(pie_data, values='Count', names='Severity', hole=0.4,
                         color='Severity',
                         color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'})
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data yet. Upload and update to see distribution.")

st.divider()

# --- DATA ENTRY & CONTROLS ---
st.subheader("3. Data Entry & AI Scan")
input_col1, input_col2 = st.columns(2)

with input_col1:
    v_category = st.radio("Vehicle Category", ["Passenger Car", "EV", "Truck"], horizontal=True)
    i_focus = st.selectbox("Focus Area", ["Exterior (Body)", "Engine Compartment"])
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    b1, b2 = st.columns(2)
    update_clicked = b1.button("🔄 Update & Log", use_container_width=True)
    if b2.button("🗑️ Reset All", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_count = 0
        st.session_state.critical_count = 0
        st.rerun()

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = np.array(img)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Sensivity Adjustments
    blur_val = (15, 15) if i_focus == "Engine Compartment" else (7, 7)
    blur = cv2.GaussianBlur(gray, blur_val, 0)
    edges = cv2.Canny(blur, 40, 130)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tek bir araç için baskın hasarı bulma
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            if area > max_area: max_area = area
            # Görselleştirme (Hepsi çizilsin ama sadece en büyüğü sonucu belirlesin)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(bgr, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # Karar Mekanizması (Tek Araç = Tek Değer)
    crit_limit = 45000 if i_focus == "Engine Compartment" else 25000
    if max_area == 0: current_sev = "None"
    elif max_area < 8000: current_sev = "Minor"
    elif max_area < crit_limit: current_sev = "Moderate"
    else: current_sev = "Critical"

    # Sonucu ekranda göster
    res_v = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with input_col2:
        st.image(res_v, use_container_width=True)
        if current_sev != "None":
            st.markdown(f"### AI Verdict: **{current_sev}**")

    # --- UPDATE LOGIC ---
    if update_clicked and current_sev != "None":
        st.session_state.total_count += 1
        if current_sev == "Critical":
            st.session_state.critical_count += 1
        
        # Geçmişe ekle (Pie chart bunu kullanacak)
        st.session_state.history.append({
            'hour': datetime.now().hour,
            'sev': current_sev
        })
        st.toast(f"Logged as {current_sev}!", icon="📍")
        st.rerun()