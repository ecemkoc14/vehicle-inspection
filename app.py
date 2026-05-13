import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Smart Inspection Dashboard", layout="wide")

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = [] 
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'critical_count' not in st.session_state:
    st.session_state.critical_count = 0

# --- HEADER & KPI ---
st.title("Vehicle Inspection & Network Analytics")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Inspected", st.session_state.total_count)
k2.metric("Critical Cases", st.session_state.critical_count)
k3.metric("Avg. Cycle Time", "12.4 min" if st.session_state.total_count > 0 else "0")
k4.metric("System Efficiency", "98%" if st.session_state.total_count > 0 else "100%")
st.divider()

# --- ANALYTICS ---
g1, g2 = st.columns(2)
with g1:
    st.subheader("Hourly Traffic Density")
    all_h = list(range(0, 24))
    counts = [0] * 24
    for item in st.session_state.history: counts[item['hour']] += 1
    fig_l = px.line(x=all_h, y=counts, markers=True)
    fig_l.update_traces(line_color='#FF4B4B')
    cur_h = datetime.now().hour
    fig_l.update_xaxes(range=[max(0, cur_h-4), min(23, cur_h+4)])
    st.plotly_chart(fig_l, use_container_width=True)

with g2:
    st.subheader("Cumulative Severity Distribution")
    if st.session_state.history:
        df_h = pd.DataFrame(st.session_state.history)
        p_data = df_h['sev'].value_counts().reset_index()
        p_data.columns = ['Severity', 'Count']
        fig_p = px.pie(p_data, values='Count', names='Severity', hole=0.4,
                     color='Severity', color_discrete_map={'Minor':'green', 'Moderate':'blue', 'Critical':'red'})
        st.plotly_chart(fig_p, use_container_width=True)
    else: st.info("No logs yet.")

st.divider()

# --- INPUT & AI SCAN ---
st.subheader("3. Data Entry & AI Scan")
col_in, col_res = st.columns(2)

with col_in:
    v_cat = st.radio("Vehicle Category", ["Passenger Car", "EV", "Truck"], horizontal=True)
    i_foc = st.selectbox("Focus Area", ["Exterior (Body)", "Engine Compartment"])
    up_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    b1, b2 = st.columns(2)
    up_btn = b1.button("🔄 Update & Log", use_container_width=True)
    if b2.button("🗑️ Reset All", use_container_width=True):
        st.session_state.history, st.session_state.total_count, st.session_state.critical_count = [], 0, 0
        st.rerun()

if up_file is not None:
    img = Image.open(up_file)
    frame = np.array(img)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # --- GELİŞMİŞ FİLTRELEME (Görüntü temizleme) ---
    # Keskin kenarları yumuşatarak kapı kolu/tekerlek gibi detayları eliyoruz
    blur = cv2.bilateralFilter(gray, 9, 75, 75) 
    
    # Dinamik Eşikleme
    thr_low = 60 if i_foc == "Engine Compartment" else 40
    edges = cv2.Canny(blur, thr_low, 150)
    
    # Morfolojik temizlik (Küçük noktaları siler)
    kernel = np.ones((5,5), np.uint8)
    processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        
        # FİLTRE: Kapı kolu veya ince uzun çizgiler (çerçeveler) elenir
        # Tekerlek gibi çok kare veya çok ince uzun olmayan, düzensiz alanları arıyoruz
        if 1500 < area < 100000 and 0.5 < aspect_ratio < 2.0:
            if area > max_area: max_area = area
            cv2.rectangle(bgr, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Karar Limitleri (Exterior için yükseltildi)
    if max_area == 0: cur_sev = "Minor" # Hiç büyük hasar yoksa
    elif max_area < 25000: cur_sev = "Minor"
    elif max_area < 55000: cur_sev = "Moderate"
    else: cur_sev = "Critical"

    with col_res:
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        c_map = {"Minor": "green", "Moderate": "blue", "Critical": "red"}
        st.markdown(f"### AI Verdict: :{c_map[cur_sev]}[{cur_sev}]")

    if up_btn:
        st.session_state.total_count += 1
        if cur_sev == "Critical": st.session_state.critical_count += 1
        st.session_state.history.append({'hour': datetime.now().hour, 'sev': cur_sev})
        st.rerun()