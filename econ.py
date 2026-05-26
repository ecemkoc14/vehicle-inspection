import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ==========================================
# 1. VERİ SETİ GENERATÖRÜ (Eğer dosya yoksa otomatik üretir)
# ==========================================
def generate_project_dataset():
    np.random.seed(42)
    n_samples = 1500
    vehicle_ids = [f"TR-{np.random.randint(10,81)}{np.random.choice(['A','B','C','D'])}{np.random.choice(['A','B','C','D'])}{np.random.randint(100,999)}" for _ in range(n_samples)]
    vehicle_types = np.random.choice(['ICE', 'EV'], size=n_samples, p=[0.7, 0.3])
    ages = np.random.randint(1, 15, size=n_samples)
    
    # Randevu ve Süre Durumları (Yeni Eklenen Modüller)
    appointment_statuses = np.random.choice(['Scheduled_OnTime', 'No_Show', 'No_Appointment_Unlawful'], size=n_samples, p=[0.75, 0.10, 0.15])
    tolerance_days_left = np.random.randint(-15, 7, size=n_samples)
    
    data = {
        'Vehicle_ID': vehicle_ids,
        'Vehicle_Type': vehicle_types,
        'Age': ages,
        'Appointment_Status': appointment_statuses,
        'Tolerance_Days_Left': tolerance_days_left,
        'CO2_Emissions': [np.nan] * n_samples,
        'NOx_Emissions': [np.nan] * n_samples,
        'OBD_Emission_Status': ['Normal'] * n_samples,
        'AdBlue_Emulator_Detected': [0] * n_samples,
        'ECU_Remap_Flag': [0] * n_samples,
        'Insulation_Resistance_M_Ohm': [np.nan] * n_samples,
        'Battery_SoH': [np.nan] * n_samples,
        'Max_Cell_Temp_C': [np.nan] * n_samples,
        'BMS_Status': ['Normal'] * n_samples,
        'Brake_Efficiency_Pct': np.random.uniform(55, 95, size=n_samples),
        'Inspection_Result': ['Pass'] * n_samples
    }
    
    df = pd.DataFrame(data)
    
    # ICE verilerini doldur ve Emisyon Hilesi (Fraud) enjekte et
    ice_mask = df['Vehicle_Type'] == 'ICE'
    n_ice = ice_mask.sum()
    df.loc[ice_mask, 'CO2_Emissions'] = 110 + df.loc[ice_mask, 'Age'] * 7 + np.random.normal(0, 10, n_ice)
    df.loc[ice_mask, 'NOx_Emissions'] = 45 + df.loc[ice_mask, 'Age'] * 4 + np.random.normal(0, 8, n_ice)
    
    fraud_ice = np.random.choice(df[ice_mask].index, size=int(n_ice * 0.12), replace=False)
    df.loc[fraud_ice, 'NOx_Emissions'] += np.random.uniform(180, 350, size=len(fraud_ice))
    df.loc[fraud_ice, 'AdBlue_Emulator_Detected'] = 1
    df.loc[fraud_ice, 'ECU_Remap_Flag'] = 1
    
    # EV verilerini doldur ve Kritik Yangın/Yalıtım Hatası enjekte et
    ev_mask = df['Vehicle_Type'] == 'EV'
    n_ev = ev_mask.sum()
    df.loc[ev_mask, 'Insulation_Resistance_M_Ohm'] = np.random.uniform(250, 900, size=n_ev)
    df.loc[ev_mask, 'Battery_SoH'] = 100 - df.loc[ev_mask, 'Age'] * np.random.uniform(1.8, 3.2, size=n_ev)
    df.loc[ev_mask, 'Max_Cell_Temp_C'] = 28 + df.loc[ev_mask, 'Age'] * 0.8 + np.random.normal(0, 2, n_ev)
    
    unsafe_ev = np.random.choice(df[ev_mask].index, size=int(n_ev * 0.09), replace=False)
    df.loc[unsafe_ev, 'Insulation_Resistance_M_Ohm'] = np.random.uniform(10, 85, size=len(unsafe_ev))
    df.loc[unsafe_ev, 'Max_Cell_Temp_C'] += np.random.uniform(25, 45, size=len(unsafe_ev))
    df.loc[unsafe_ev, 'BMS_Status'] = 'Critical_Fault'
    
    # Sonuç Kuralları
    df.loc[df['Brake_Efficiency_Pct'] < 60, 'Inspection_Result'] = 'Fail_Mechanical'
    df.loc[(df['Vehicle_Type'] == 'ICE') & ((df['NOx_Emissions'] > 170) | (df['AdBlue_Emulator_Detected'] == 1)), 'Inspection_Result'] = 'Fail_Emission_Fraud'
    df.loc[(df['Vehicle_Type'] == 'EV') & ((df['Insulation_Resistance_M_Ohm'] < 100) | (df['Max_Cell_Temp_C'] > 55)), 'Inspection_Result'] = 'Fail_EV_Safety'
    
    df.to_csv('digital_twin_vehicle_inspection_dataset.csv', index=False)
    return df

# Veri setini yükle veya oluştur
try:
    df_vehicles = pd.read_csv('digital_twin_vehicle_inspection_dataset.csv')
except FileNotFoundError:
    df_vehicles = generate_project_dataset()

# ==========================================
# 2. STREAMLIT ARAYÜZÜ VE DIJITAL İKİZ SİMÜLASYONU
# ==========================================
st.set_page_config(page_title="Araç Muayene Dijital İkiz", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🌐 Siber-Fiziksel Araç Denetim Ağlarında Dijital İkiz Platformu</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>HGS Makro Veri Akışı, Randevu/Ceza Mekanizması ve İstasyon İçi Esnek Darboğaz Yönetimi Prototipi</p>", unsafe_allow_html=True)
st.hr()

# State Tanımlamaları
if 'ice_queue' not in st.session_state: st.session_state.ice_queue = 3
if 'ev_queue' not in st.session_state: st.session_state.ev_queue = 8
if 'hybrid_mode' not in st.session_state: st.session_state.hybrid_mode = False
if 'logs' not in st.session_state: st.session_state.logs = []

# Ana Ekran Düzeni (3 Bölüm)
col1, col2, col3 = st.columns([1.2, 1, 1])

# --- SOL SÜTUN: HGS MAKRO AKIŞI (EMÜLATÖR) ---
with col1:
    st.header("🛣️ HGS Gişe Canlı Akış Emülatörü")
    st.write("Otoyol antenlerinden Dijital İkiz bulutuna düşen anlık plaka ve süre verileri:")
    
    run_hgs = st.button("HGS Canlı Akışını Tetikle", key="hgs_btn")
    
    if run_hgs:
        st.session_state.logs = [] # Logları temizle
        sample_batch = df_vehicles.sample(6)
        
        for idx, row in sample_batch.iterrows():
            v_id = row['Vehicle_ID']
            v_type = row['Vehicle_Type']
            app_status = row['Appointment_Status']
            days_left = row['Tolerance_Days_Left']
            
            # Randevu ve Kural Kontrol Motoru
            if app_status == 'No_Appointment_Unlawful' and days_left < 0:
                log_msg = f"❌ {v_id} ({v_type}) HGS Geçişi: Muayene Süresi {abs(days_left)} Gün Geçmiş! Randevu Alınmadığı İçin Otomatik Elektronik Ceza Kesildi."
                st.error(log_msg)
                st.session_state.logs.append(("error", log_msg))
            elif app_status == 'No_Show':
                log_msg = f"⚠️ {v_id} ({v_type}) HGS Geçişi: Randevusuna Gitmeyip Kapasite Bloke Ettiği Tespit Edildi! No-Show Cezası Uygulandı."
                st.warning(log_msg)
                st.session_state.logs.append(("warning", log_msg))
            else:
                log_msg = f"✅ {v_id} ({v_type}) HGS Geçişi: Randevusu Planlanmış Zaman Diliminde (Kalan Gün: {days_left}). Tolerans Dahilinde."
                st.success(log_msg)
                st.session_state.logs.append(("success", log_msg))
                
                # İstasyona araç ekle (Simülasyon kuyruğunu dinamik besle)
                if v_type == 'ICE': st.session_state.ice_queue += 1
                else: st.session_state.ev_queue += 1
            time.sleep(0.4)

# --- ORTA SÜTUN: MİKRO İSTASYON DARBOĞAZ SİMÜLASYONU ---
with col2:
    st.header("🏢 İstasyon İçi Mikro Kuyruk")
    st.write("HGS'den gelen talebe göre anlık oluşan kuyruk ve darboğaz analizi:")
    
    # Darboğaz durum hesaplaması
    ice_wait = st.session_state.ice_queue * 11 / 2 # 2 adet standart hat var
    ev_wait = st.session_state.ev_queue * 16 / (2 if st.session_state.hybrid_mode else 1) # Esnek hat devredeyse böl jeneratörünü
    
    st.metric(label="ICE (Konvansiyonel) Kuyruğu", value=f"{st.session_state.ice_queue} Araç", delta=f"Bekleme ~{ice_wait:.1f} dk")
    st.metric(label="EV (Elektrikli) Kuyruğu", value=f"{st.session_state.ev_queue} Araç", delta=f"Bekleme ~{ev_wait:.1f} dk (KRİTİK BOTTLEMECK)", delta_color="inverse" if ev_wait > 30 else "normal")
    
    st.subheader("🤖 Dijital İkiz Karar Destek Kararı")
    if st.session_state.ev_queue >= 7 and not st.session_state.hybrid_mode:
        st.error("⚠️ DARBOĞAZ ALARMI: EV hattı bekleme süresi kritik sınırı aştı! Boşta kalan Hat-2'nin hibrit moda geçirilmesi öneriliyor.")
        if st.button("Hat-2'yi Esnek Hibrit Moda Geçir"):
            st.session_state.hybrid_mode = True
            st.rerun()
    elif st.session_state.hybrid_mode:
        st.success("🔄 Akıllı Hibrit Mod Aktif: Hat-2 şu an hem ICE hem EV testi yapabiliyor. Kuyruk erime hızı %100 arttı.")
        if st.button("Sistem Normale Dön / Esnek Modu Kapat"):
            st.session_state.hybrid_mode = False
            st.rerun()
            
    if st.button("Kuyruğu Erit (Simülasyonu İlerlet)"):
        if st.session_state.ice_queue > 0: st.session_state.ice_queue -= np.random.randint(1,3)
        if st.session_state.ev_queue > 0: st.session_state.ev_queue -= (np.random.randint(2,4) if st.session_state.hybrid_mode else 1)
        st.session_state.ice_queue = max(0, st.session_state.ice_queue)
        st.session_state.ev_queue = max(0, st.session_state.ev_queue)
        st.rerun()

# --- SAĞ SÜTUN: EV GÜVENLİĞİ VE EMİSYON MANİPÜLASYON ANALİTİĞİ ---
with col3:
    st.header("🔬 İleri Teknolojik Muayene Odası")
    st.write("Hatta giren rastgele bir aracın siber-fiziksel test analizör sonuçları:")
    
    test_btn = st.button("Rastgele Araç Test Et")
    if test_btn:
        test_car = df_vehicles.sample(1).iloc[0]
        st.markdown(f"**Test Edilen Plaka:** `{test_car['Vehicle_ID']}`")
        st.markdown(f"**Araç Sınıfı:** `{test_car['Vehicle_Type']}`")
        
        if test_car['Vehicle_Type'] == 'ICE':
            st.info("📊 Egzoz Gaz Analizörü & OBD Çapraz Kontrolü")
            st.write(f"Fiziksel NOx Salınımı: {test_car['NOx_Emissions']:.2f} ppm")
            st.write(f"OBD Arıza Beyanı: {test_car['OBD_Emission_Status']}")
            
            if test_car['Inspection_Result'] == 'Fail_Emission_Fraud':
                st.error("🚨 MANİPÜLASYON DETECTED! OBD 'Sorun Yok' bildiriyor ancak fiziksel emisyon yasal sınırın üstünde. AdBlue Emülatörü/Yazılım Hilesi Saptandı. Muayene İptal!")
            else:
                st.success("✅ Emisyon Değerleri ve Beyin Yazılımı Yasal Sınırlarda.")
                
        else: # EV
            st.info("⚡ Yüksek Voltaj İzolasyon & Termal Batarya Taraması")
            st.write(f"Yalıtım Direnci: {test_car['Insulation_Resistance_M_Ohm']:.1f} MΩ")
            st.write(f"Max Hücre Sıcaklığı: {test_car['Max_Cell_Temp_C']:.1f} °C")
            
            if test_car['Inspection_Result'] == 'Fail_EV_Safety':
                st.error("🚨 EV SAFETY HAZARD! Kritik İzolasyon Sızıntısı veya Batarya Hücrelerinde Aşırı Isınma (Thermal Runaway Riski). Araç Trafikten Men Edildi!")
            else:
                st.success("✅ Batarya Sağlığı (SoH) ve Yüksek Voltaj Yalıtım Güvenliği Doğrulandı.")

st.hr()
# Alt Grafik Alanı (Mevcut Veri Setinin Dağılımları)
st.subheader("📈 Veri Seti Kusur / Başarı Dağılım Grafikleri (Jüri Analiz Paneli)")
c1, c2 = st.columns(2)
with c1:
    fig1 = px.histogram(df_vehicles, x="Inspection_Result", color="Vehicle_Type", title="Araç Tipine Göre Muayene Red/Kabul Dağılımları", barmode="group", color_discrete_sequence=["#1E3A8A", "#10B981"])
    st.plotly_chart(fig1, use_container_width=True)
with c2:
    fig2 = px.scatter(df_vehicles[df_vehicles['Vehicle_Type']=='EV'], x="Insulation_Resistance_M_Ohm", y="Max_Cell_Temp_C", color="Inspection_Result", title="Elektrikli Araçlarda İzolasyon Direnci ve Sıcaklık Korelasyonu")
    st.plotly_chart(fig2, use_container_width=True)