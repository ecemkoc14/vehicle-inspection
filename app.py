import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import datetime
import time

# --- SESSION STATE: VERİLERİ SAKLAMA ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['ID', 'Type', 'Severity', 'Time'])
if 'hourly_counts' not in st.session_state:
    # 08:00 - 20:00 arası boş bir trafik tablosu
    st.session_state.hourly_counts = {f"{h:02d}:00": 0 for h in range(8, 21)}

st.set_page_config(page_title="VisionControl v3", layout="wide")

st.title("🛡️ VisionControl: Intelligent Inspection Simulator")
st.markdown("##### *Dynamic Queue Management & Context-Aware AI Analysis*")

# --- ÜST PANEL: METRİKLER ---
m1, m2, m3, m4 = st.columns(4)
total_v = len(st.session_state.history)
m1.metric("Total Vehicles Today", total_v)
m2.metric("Critical Alerts", len(st.session_state.history[st.session_state.history['Severity'] == 'CRITICAL']))
m3.metric("System Efficiency", "94%", "+2%")
m4.metric("Avg. Wait Time", f"{random.randint(10, 15)} min" if total_v > 0 else "0 min")

st.divider()

# --- GRAFİKLER (DİNAMİK) ---
g1, g2 = st.columns(2)

with g1:
    # Saatlik yoğunluk grafiği artık st.session_state.hourly_counts verisinden besleniyor
    df_traffic = pd.DataFrame(list(st.session_state.hourly_counts.items()), columns=['Hour', 'Vehicles'])
    fig_traffic = px.area(df_traffic, x='Hour', y='Vehicles', title="Live Hourly Traffic (Dynamic)", markers=True)
    st.plotly_chart(fig_traffic, use_container_width=True)

with g2:
    # Hasar dağılım grafiği
    if not st.session_state.history.empty:
        fig_sev = px.pie(st.session_state.history, names='Severity', title="Daily Severity Distribution", hole=0.4)
        st.plotly_chart(fig_sev, use_container_width=True)
    else:
        st.info("No data yet to generate severity chart.")

st.divider()

# --- ANALİZ VE SİMÜLASYON ---
st.subheader("🤖 Smart Scanning & Lane Assignment")
i1, i2 = st.columns([1, 1])

with i1:
    v_type = st.selectbox("Vehicle Class", ["Passenger", "Electric (EV)", "Truck"])
    inspection_focus = st.selectbox("Inspection Focus", ["Exterior (Body Surface)", "Engine/Battery Compartment"])
    up_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if st.button("Reset Global Data"):
        st.session_state.history = pd.DataFrame(columns=['ID', 'Type', 'Severity', 'Time'])
        st.session_state.hourly_counts = {f"{h:02d}:00": 0 for h in range(8, 21)}
        st.rerun()

with i2:
    if up_file:
        img = Image.open(up_file)
        frame = np.array(img)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # --- ÖZELLEŞTİRİLMİŞ AI ANALİZİ ---
        # Kaput içi ise hassasiyeti azaltıyoruz (False Positive engellemek için)
        if inspection_focus == "Engine/Battery Compartment":
            blur_val = (21, 21) # Daha fazla bulanıklaştırma (detayları siler)
            canny_low, canny_high = 80, 180 # Daha yüksek eşik (sadece çok net hasarları görür)
            min_area = 3500 # Daha büyük bir alan hasar sayılır
        else:
            blur_val = (9, 9)
            canny_low, canny_high = 40, 110
            min_area = 800

        blur = cv2.GaussianBlur(gray, blur_val, 0)
        edges = cv2.Canny(blur, canny_low, canny_high)
        
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        damage_total = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                damage_total += area
                cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

        st.image(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("Approve Analysis & Route Vehicle"):
            # Hasar seviyesi belirleme
            if damage_total > 35000: sev = "CRITICAL"
            elif damage_total > 8000: sev = "MODERATE"
            else: sev = "MINOR"
            
            # Veriyi kaydet
            new_id = np.random.randint(1000, 9999)
            now = datetime.datetime.now()
            current_hour = now.strftime("%H:00")
            
            # Geçmişe ekle
            new_entry = pd.DataFrame({'ID': [new_id], 'Type': [v_type], 'Severity': [sev], 'Time': [current_hour]})
            st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
            
            # Grafiği güncelle (Eğer çalışma saatleri içindeyse)
            if current_hour in st.session_state.hourly_counts:
                st.session_state.hourly_counts[current_hour] += 1
            
            st.success(f"Vehicle #{new_id} assigned. System Status: {sev}")
            time.sleep(0.5)
            st.rerun()