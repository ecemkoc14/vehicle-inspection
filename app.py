!pip install -q streamlit opencv-python-headless numpy Pillow
import urllib.request
with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
import cv2
import numpy as np
from PIL import Image
st.set_page_config(page_title="Inspection System", layout="wide")
st.title("Vehicle Inspection and Safety Assessment Dashboard")
input_source = st.file_uploader("Upload vehicle engine compartment data", type=["jpg", "jpeg", "png"])
if input_source is not None:
    raw_image = Image.open(input_source)
    matrix_data = np.array(raw_image)
    processing_frame = cv2.cvtColor(matrix_data, cv2.COLOR_RGB2BGR)
    grayscale_layer = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
    denoised_data = cv2.GaussianBlur(grayscale_layer, (7, 7), 0)
    segmentation_threshold = cv2.threshold(denoised_data, 140, 255, cv2.THRESH_BINARY_INV)[1]
    feature_boundaries, _ = cv2.findContours(segmentation_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anomaly_counter = 0
    for segment in feature_boundaries:
        if cv2.contourArea(segment) > 650:
            (pos_x, pos_y, rect_w, rect_h) = cv2.boundingRect(segment)
            cv2.rectangle(processing_frame, (pos_x, pos_y), (pos_x + rect_w, pos_y + rect_h), (255, 0, 0), 2)
            anomaly_counter += 1
    output_visualization = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
    primary_col, secondary_col = st.columns(2)
    with primary_col:
        st.image(raw_image, caption="Original Input Data", use_container_width=True)
    with secondary_col:
        st.image(output_visualization, caption="Automated Detection Output", use_container_width=True)
    st.divider()
    if anomaly_counter > 0:
        st.warning(f"Detection Report: {anomaly_counter} potential risk zones identified.")
    else:
        st.success("Assessment Complete: No significant technical irregularities detected.")
    st.sidebar.subheader("System Metadata")
    st.sidebar.text("Status: Active")
    st.sidebar.text("Module: Visual Inspection")
''')
print("1. Copy this IP address:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())
print("2. Click the link below and paste the IP into 'Endpoint IP' field.")
!npx localtunnel --port 8501 & streamlit run app.py