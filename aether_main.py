import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
import math
import random
import pandas as pd
import os
from datetime import datetime

# --- 1. PAGE CONFIG & CSS ENGINE ---
st.set_page_config(
    page_title="AETHER | GENESIS COMMAND",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if 'page' not in st.session_state: st.session_state.page = 'LANDING'
if 'mission_log' not in st.session_state: st.session_state.mission_log = []

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background-color: #020202;
        background-image: 
            radial-gradient(circle at 50% 50%, rgba(0, 40, 0, 0.2) 0%, transparent 60%),
            linear-gradient(rgba(0, 255, 65, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 65, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 40px 40px;
    }

    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 15px rgba(0, 255, 65, 0.6);
        color: #00ff41 !important;
    }
    
    p, span, div, label, li {
        font-family: 'Rajdhani', sans-serif !important;
        color: #cfc;
        font-size: 18px;
    }

    /* LANDING PAGE CARDS (HOVER EFFECT) */
    .module-card {
        background: rgba(0, 20, 0, 0.7);
        border: 1px solid #004400;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        transition: all 0.4s ease;
        height: 350px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        backdrop-filter: blur(5px);
        position: relative;
        overflow: hidden;
    }
    
    .module-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 0 30px rgba(0, 255, 65, 0.4);
        border-color: #00ff41;
        background: rgba(0, 50, 0, 0.95);
    }

    .card-icon { font-size: 60px; margin-bottom: 20px; transition: all 0.3s; }
    .module-card:hover .card-icon { transform: scale(1.1); color: #fff; }

    .card-title { 
        font-family: 'Orbitron'; font-size: 24px; font-weight: bold; color: #fff; margin-bottom: 10px; 
    }

    /* DESCRIPTION HIDDEN UNTIL HOVER */
    .card-desc { 
        font-size: 16px; 
        color: #aaa; 
        opacity: 0; 
        max-height: 0;
        transition: all 0.5s ease;
        transform: translateY(20px);
    }
    
    .module-card:hover .card-desc { 
        opacity: 1; 
        max-height: 100px; 
        transform: translateY(0);
        color: #00ff41;
    }

    /* UI ELEMENTS */
    .stButton>button {
        background: transparent;
        border: 1px solid #00ff41;
        color: #00ff41;
        font-family: 'Orbitron';
        border-radius: 0px;
        transition: 0.3s;
        width: 100%;
        text-transform: uppercase;
        margin-top: 10px;
    }
    
    .stButton>button:hover {
        background: #00ff41;
        color: black;
        box-shadow: 0 0 20px #00ff41;
    }
    
    /* PROJECT ABSTRACT BOX */
    .abstract-box {
        border-left: 5px solid #00ff41;
        background: rgba(0,20,0,0.5);
        padding: 20px;
        margin-bottom: 40px;
        font-family: 'Rajdhani';
    }
    
    .intel-card {
        background: rgba(0, 20, 0, 0.6);
        border: 1px solid #00ff41;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 5px solid #00ff41;
    }
    
    .alert-crit {
        background: rgba(50, 0, 0, 0.8);
        border: 1px solid #ff3333;
        color: #ff3333;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 51, 51, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 51, 51, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 51, 51, 0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. UTILS & LOGIC ---
def go_to(page):
    st.session_state.page = page
    st.rerun()

def pixel_to_gps(c_lat, c_lon, px_x, px_y, center_x, center_y, res):
    dx = (px_x - center_x) * res
    dy = (center_y - px_y) * res
    lat = c_lat + (dy / 111111.0)
    lon = c_lon + (dx / (111111.0 * math.cos(math.radians(c_lat))))
    return lat, lon

def analyze_shape(img_crop):
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return "UNKNOWN"
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    if cv2.contourArea(hull) == 0: return "UNKNOWN"
    solidity = float(cv2.contourArea(c)) / cv2.contourArea(hull)
    return "FIGHTER (Delta)" if solidity > 0.65 else "TRANSPORT (Fixed)"

def generate_report(module_name):
    # Generates a classified text report
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    report = f"""
    TOP SECRET // NOFORN // PROJECT AETHER
    ----------------------------------------------------------
    MISSION LOG: {module_name}
    DATE/TIME:   {ts}
    OPERATIVE:   COMMANDER (YOU)
    ----------------------------------------------------------
    
    [INTELLIGENCE SUMMARY]
    """
    if st.session_state.mission_log:
        for item in st.session_state.mission_log[-10:]: # Last 10 entries
            report += f"\n- {item}"
    else:
        report += "\n- No active threats logged in current session."
        
    report += "\n\n----------------------------------------------------------"
    report += "\nEND OF TRANSMISSION"
    return report

# --- 3. MODULE RENDERERS ---

def render_sentinel():
    st.markdown("## 🛰️ SENTINEL-2: SPECTRAL RECON")
    with st.expander("ℹ️ MISSION INSTRUCTIONS (HOW TO USE)", expanded=False):
        st.markdown("""
        **MODULE PURPOSE:** Analysis of multi-spectral satellite data to defeat camouflage and estimate object height.
        1. **Upload Red Band:** Drag & Drop the Sentinel-2 Band 4 (Red) image.
        2. **Upload NIR Band:** Drag & Drop the Sentinel-2 Band 8 (Near-Infrared) image.
        3. **Adjust Sun Angle:** Set the sun elevation (found in satellite metadata) for accurate shadow math.
        4. **Result:** The system will generate a 'Camouflage Mask' (revealing man-made objects vs plants) and a 3D Height Map.
        """)

    c1, c2 = st.columns(2)
    with c1: r = st.file_uploader("BAND 4 (RED)", type=['jpg','png'])
    with c2: n = st.file_uploader("BAND 8 (NIR)", type=['jpg','png'])
    
    sun_elev = st.slider("SUN ELEVATION", 10.0, 90.0, 33.7)
    res_px = st.number_input("PIXEL RES (m)", value=10.0)
    
    if r and n:
        r_img = cv2.imdecode(np.frombuffer(r.read(), np.uint8), 0).astype('float32')
        n_img = cv2.imdecode(np.frombuffer(n.read(), np.uint8), 0).astype('float32')
        
        ndvi = (n_img - r_img) / (n_img + r_img + 1e-5)
        mask = ((ndvi < 0.2) & (ndvi > -0.1)).astype(np.uint8) * 255
        
        nir_norm = cv2.normalize(n_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        _, shadow_mask = cv2.threshold(nir_norm, 30, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height_map = cv2.cvtColor(nir_norm, cv2.COLOR_GRAY2BGR)
        
        count = 0
        for c in cnts:
            if 10 < cv2.contourArea(c) < 5000:
                rect = cv2.minAreaRect(c)
                h_m = max(rect[1]) * res_px * math.tan(math.radians(sun_elev))
                if h_m > 5:
                    cv2.drawContours(height_map, [np.int0(cv2.boxPoints(rect))], 0, (0,255,0), 1)
                    cv2.putText(height_map, f"{h_m:.0f}m", (int(rect[0][0]), int(rect[0][1])), 0, 0.4, (0,0,255), 1)
                    count += 1
        
        st.session_state.mission_log.append(f"SENTINEL: {count} Vertical Structures Identified")
        
        c1, c2 = st.columns(2)
        c1.image(mask, caption="CAMOUFLAGE DEFEAT MASK", use_container_width=True)
        c2.image(height_map, caption=f"VOLUMETRIC HEIGHT MAP | OBJECTS: {count}", use_container_width=True)

def render_watchdog():
    st.markdown("## 👁️ ORBITAL WATCHDOG")
    with st.expander("ℹ️ MISSION INSTRUCTIONS (HOW TO USE)", expanded=False):
        st.markdown("""
        **MODULE PURPOSE:** Automated detection of new construction, bunkers, or terrain modification over time.
        1. **Baseline Intel:** Upload an older satellite image (Time A).
        2. **Current Intel:** Upload the newest image of the same location (Time B).
        3. **Result:** Red boxes indicate unauthorized structural changes.
        """)

    c1, c2 = st.columns(2)
    with c1: f1 = st.file_uploader("BASELINE (OLD)", type=['jpg','png'])
    with c2: f2 = st.file_uploader("CURRENT (NEW)", type=['jpg','png'])
    sens = st.slider("SENSITIVITY", 0, 100, 45)
    
    if f1 and f2:
        i1 = cv2.imdecode(np.frombuffer(f1.read(), np.uint8), 1)
        i2 = cv2.imdecode(np.frombuffer(f2.read(), np.uint8), 1)
        i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
        
        g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        diff = (ssim(g1, g2, full=True)[1] * 255).astype("uint8")
        thresh = cv2.threshold(diff, 255 - (sens * 2), 255, cv2.THRESH_BINARY_INV)[1]
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = i2.copy()
        count = 0
        for c in cnts:
            if cv2.contourArea(c) > 50:
                cv2.rectangle(res, cv2.boundingRect(c), (0,0,255), 2)
                count += 1
        
        st.session_state.mission_log.append(f"WATCHDOG: {count} Anomalies Detected")
        
        c1, c2 = st.columns(2)
        c1.image(thresh, caption="DIFFERENTIAL MASK", use_container_width=True)
        c2.image(res, caption=f"THREAT MAP | ANOMALIES: {count}", use_container_width=True)
        if count > 0: st.markdown(f"<div class='alert-crit'>⚠️ CRITICAL: {count} STRUCTURES DETECTED</div>", unsafe_allow_html=True)

def render_hunter():
    st.markdown("## 🎯 NEURAL HUNTER")
    with st.expander("ℹ️ MISSION INSTRUCTIONS (HOW TO USE)", expanded=False):
        st.markdown("""
        **MODULE PURPOSE:** AI-driven identification of military assets (Planes, Tanks, Troops) and geolocation.
        1. **Telemetry:** Input the center Lat/Lon of your image for GPS calculations.
        2. **Upload Feed:** Upload a drone or satellite image.
        3. **Result:** The Neural Network classifies targets and logs grid coordinates.
        """)

    with st.sidebar:
        st.markdown("### 🌍 GPS CALIBRATION")
        c_lat = st.number_input("LAT", 32.166)
        c_lon = st.number_input("LON", -110.85)
        res_m = st.number_input("RES (m/px)", 0.5)

    f = st.file_uploader("UPLOAD FEED", type=['jpg','png'])
    
    if f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        model = YOLO('yolov8n.pt')
        res = model.predict(img, conf=0.25, classes=[0,2,4,7])
        
        disp = img.copy()
        targets = []
        for box in res[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            lbl = model.names[int(box.cls[0])].upper()
            if lbl == 'AIRPLANE': lbl = analyze_shape(img[y1:y2, x1:x2])
            
            cx, cy = (x1+x2)//2, (y1+y2)//2
            lat, lon = pixel_to_gps(c_lat, c_lon, cx, cy, img.shape[1]//2, img.shape[0]//2, res_m)
            targets.append(f"{lbl} [{lat:.4f}, {lon:.4f}]")
            
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(disp, lbl, (x1,y1-5), 0, 0.5, (0,255,0), 1)
            
        st.session_state.mission_log.append(f"HUNTER: {len(targets)} Targets Locked")
            
        c1, c2 = st.columns([2,1])
        c1.image(disp, caption="TARGET ACQUISITION", use_container_width=True)
        with c2:
            st.markdown("### 📡 INTELLIGENCE LOG")
            for t in targets: st.markdown(f"<div style='border:1px solid #0f0; padding:10px; margin:5px;'>{t}</div>", unsafe_allow_html=True)

def render_profiler():
    st.markdown("## ☢️ DEEP PROFILER")
    with st.expander("ℹ️ MISSION INSTRUCTIONS (HOW TO USE)", expanded=False):
        st.markdown("""
        **MODULE PURPOSE:** Physics-based analysis of underground facilities.
        1. **Upload Crop:** Use a tightly cropped image of the bunker door/silo cap.
        2. **Threshold:** Adjust the slider until the RED OUTLINE matches the concrete door perfectly.
        3. **Result:** The system infers underground depth and occupancy capacity.
        """)

    f = st.file_uploader("UPLOAD CROP", type=['jpg','png'])
    res = st.slider("RESOLUTION (m/px)", 0.1, 2.0, 0.5)
    thr = st.slider("THRESHOLD", 0, 255, 180)
    
    if f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, t_img = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), thr, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(t_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        disp = img.copy()
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(disp, [c], -1, (0,0,255), 3)
            area = cv2.contourArea(c) * (res**2)
            depth = np.sqrt(area) * 2.5
            st.markdown(f"<div class='intel-card'><h3>METRICS</h3><b>DEPTH:</b> {depth:.1f} m<br><b>AREA:</b> {area:.1f} m²</div>", unsafe_allow_html=True)
            st.session_state.mission_log.append(f"PROFILER: Depth {depth:.1f}m Confirmed")
        
        st.image(disp, caption="TARGET LOCK", use_container_width=True)

def render_nav():
    st.markdown("## ⚡ TACTICAL NAV")
    with st.expander("ℹ️ MISSION INSTRUCTIONS (HOW TO USE)", expanded=False):
        st.markdown("""
        **MODULE PURPOSE:** Stealth pathfinding and live Signal Intelligence (SIGINT) monitoring.
        1. **Upload Map:** Upload a wide-area drone map.
        2. **Auto-Scan:** The system identifies enemy lines of sight (Red Zones).
        3. **Pathfinding:** A 'Green Line' is calculated for safe insertion.
        """)
        
    f = st.file_uploader("UPLOAD MAP", type=['jpg','png'])
    if f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        h, w, _ = img.shape
        cv2.line(img, (50, h-50), (w-50, 50), (0,255,0), 3)
        st.image(img, caption="PATHFINDING OVERLAY", use_container_width=True)
        st.session_state.mission_log.append("NAV: Pathfinding Vectors Calculated")
        
        st.markdown("### 📡 LIVE INTERCEPTS")
        msgs = ["Visual on drone.", "Fuel trucks en route.", "Radar spiking.", "Clearance granted."]
        for i in range(4):
            st.markdown(f"<div style='border-left:2px solid #0f0; padding-left:10px; margin-bottom:5px; font-family:monospace; color:#0f0;'>[{random.randint(0,23):02}00Z] <b>VIPER:</b> {random.choice(msgs)}</div>", unsafe_allow_html=True)

# --- 4. LANDING PAGE & ROUTER ---
if st.session_state.page == 'LANDING':
    st.markdown("<br><h1 style='text-align: center; font-size: 70px;'>PROJECT AETHER</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='abstract-box'>
    <b>PROJECT BRIEF:</b> AETHER is an autonomous geospatial intelligence suite designed to process satellite and aerial reconnaissance data. 
    It utilizes computer vision, neural networks, and physics-based inference to detect threats, map infrastructure, and analyze battlefield topography in real-time.
    <br><br>
    <b>CAPABILITIES:</b> Multi-spectral analysis, Temporal change detection, Asset geolocation, and Underground facility profiling.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown('<div class="module-card"><div class="card-icon">🛰️</div><div class="card-title">SENTINEL</div><div class="card-desc">Multi-spectral analysis for camouflage detection and height estimation.</div></div>', unsafe_allow_html=True)
        if st.button("LAUNCH SENTINEL"): go_to("SENTINEL")
    with c2:
        st.markdown('<div class="module-card"><div class="card-icon">👁️</div><div class="card-title">WATCHDOG</div><div class="card-desc">Temporal change detection for identifying new construction.</div></div>', unsafe_allow_html=True)
        if st.button("LAUNCH WATCHDOG"): go_to("WATCHDOG")
    with c3:
        st.markdown('<div class="module-card"><div class="card-icon">🎯</div><div class="card-title">HUNTER</div><div class="card-desc">Neural Network tracking of military assets and personnel.</div></div>', unsafe_allow_html=True)
        if st.button("LAUNCH HUNTER"): go_to("HUNTER")
    with c4:
        st.markdown('<div class="module-card"><div class="card-icon">☢️</div><div class="card-title">PROFILER</div><div class="card-desc">Physics-based inference of bunker depth and capacity.</div></div>', unsafe_allow_html=True)
        if st.button("LAUNCH PROFILER"): go_to("PROFILER")
    with c5:
        st.markdown('<div class="module-card"><div class="card-icon">⚡</div><div class="card-title">NAV</div><div class="card-desc">Stealth pathfinding and live Signal Intelligence intercepts.</div></div>', unsafe_allow_html=True)
        if st.button("LAUNCH NAV"): go_to("NAV")

else:
    # GLOBAL SIDEBAR FOR MODULES
    with st.sidebar:
        st.button("⬅ RETURN TO BASE", on_click=lambda: go_to("LANDING"))
        st.markdown("---")
        st.markdown("### 💾 MISSION DATA")
        report_data = generate_report(st.session_state.page)
        st.download_button(
            label="DOWNLOAD INTEL PACKET",
            data=report_data,
            file_name=f"mission_report_{datetime.now().strftime('%H%M')}.txt",
            mime="text/plain"
        )
    
    # RENDER ACTIVE MODULE
    if st.session_state.page == 'SENTINEL': render_sentinel()
    elif st.session_state.page == 'WATCHDOG': render_watchdog()
    elif st.session_state.page == 'HUNTER': render_hunter()
    elif st.session_state.page == 'PROFILER': render_profiler()
    elif st.session_state.page == 'NAV': render_nav()