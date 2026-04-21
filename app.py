import streamlit as st
import torch
import numpy as np
import glob
import os
from utils.preprocess import load_image_from_bytes, resize_for_display, bgr_to_rgb
from utils.dehaze import dehaze
from utils.inference import LaneDetector, overlay_unet_mask

# Page Config
st.set_page_config(page_title="ADAS Vision Pipeline", layout="wide", page_icon="🛣️")

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        /* Global Font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main background */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #1a1f2e 0%, #0a0a0a 100%);
        }
        
        /* Headers */
        h1.main-title {
            background: -webkit-linear-gradient(45deg, #00FF87, #60EFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            text-align: center;
            padding-bottom: 5px;
            margin-bottom: 0px;
        }
        
        .sub-title {
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 40px;
        }
        
        /* File Uploader styling */
        [data-testid="stFileUploadDropzone"] {
            border: 2px dashed rgba(0, 255, 135, 0.5) !important;
            border-radius: 16px !important;
            background: rgba(0, 255, 135, 0.02) !important;
            transition: all 0.3s ease;
            padding: 40px !important;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            background: rgba(0, 255, 135, 0.08) !important;
            border-color: #00FF87 !important;
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0,255,135,0.15);
        }
        
        /* Images */
        [data-testid="stImage"] img {
            border-radius: 12px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.8);
            transition: transform 0.3s ease;
            border: 1px solid rgba(255,255,255,0.05);
        }
        [data-testid="stImage"] img:hover {
            transform: scale(1.02);
        }
        
        /* Headings */
        h3 {
            color: #E2E8F0 !important;
            font-weight: 600 !important;
        }
        
        hr {
            border-color: rgba(255,255,255,0.1) !important;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_model():
    weight_files = glob.glob("weights/model_epoch_*.pth")
    if weight_files:
        latest_model = max(weight_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Loaded latest weights dynamically: {latest_model}")
        return LaneDetector(model_path=latest_model)
    return LaneDetector(model_path="weights/model.pth")

def main():
    inject_custom_css()
    
    st.markdown("<h1 class='main-title'>ADAS Vision Engine</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Deep Learning Foggy Lane Detection System</div>", unsafe_allow_html=True)
    
    model = get_model()
    
    # Center the uploader
    col_up1, col_up2, col_up3 = st.columns([1, 2, 1])
    with col_up2:
        uploaded_file = st.file_uploader("Drop diagnostic image payload here...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        image_bgr = load_image_from_bytes(uploaded_file)
        image_bgr = resize_for_display(image_bgr)
        
        # Interactive Thresholding logic inside a sleek container
        with st.container():
            st.markdown("### ⚙️ Calibration Settings")
            threshold = st.slider(
                "U-Net Signal Sensitivity (Slide left to force the model to render faint detections)", 
                min_value=0.00, max_value=0.99, value=0.50, step=0.01
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner("Dehazing image out of the fog..."):
            dehazed_bgr = dehaze(image_bgr)
            
        with st.spinner("Running U-Net Inference..."):
            prob_mask = model.predict(dehazed_bgr)
            binary_mask = (prob_mask > threshold).astype(np.uint8)
            final_output = overlay_unet_mask(dehazed_bgr, binary_mask, color=(0, 255, 135), alpha=0.5) # Neon green overlay
            
        st.divider()
        
        # Display constraints
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original Sensor Feed**")
            st.image(bgr_to_rgb(image_bgr), width='stretch')
        with col2:
            st.markdown("**Dehazed Matrix**")
            st.image(bgr_to_rgb(dehazed_bgr), width='stretch')
        with col3:
            st.markdown(f"**U-Net Detection (>{threshold*100:.0f}% Confidence)**")
            st.image(bgr_to_rgb(final_output), width='stretch')
            
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔬 Debug: Raw Model Probability Heatmap", expanded=False):
            st.markdown("This heatmap reveals exactly what the U-Net is thinking. Brighter pixels = Higher confidence.")
            st.image(prob_mask, clamp=True, width='stretch')
            
if __name__ == "__main__":
    main()
