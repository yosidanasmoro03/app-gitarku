import os
import time
import base64
import random
import threading
import numpy as np
import cv2
import av

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ==========================================
# 1. KONFIGURASI & KONSTANTA
# ==========================================

st.set_page_config(
    layout="wide", 
    page_title="App Gitarku", 
    initial_sidebar_state="collapsed"
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Daftar pertanyaan kuis
QUIZ_DATA = {
    "Bentuk jari chord C": "C-Chord",
    "Tunjukkan chord G": "G-Chord",
    "Bentuk chord D": "D-Chord",
    "Chord A di fretboard": "A-Chord",
    "Tunjukkan chord E": "E-Chord",
    "Posisi jari chord Am": "Am-Chord",
    "Tampilkan chord F": "F-Chord",
    "Bentuk chord Bm": "Bm-Chord",
    "Tunjukkan chord Dm": "Dm-Chord",
    "Bentuk chord Em": "Em-Chord"
}

# ==========================================
# 2. UTILITY FUNCTIONS (CSS, GAMBAR, MODEL)
# ==========================================

def load_css(file_path):
    """Memuat file CSS eksternal."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_img_as_base64(file_path):
    """Mengubah file gambar menjadi string base64 untuk HTML injection."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_overlay(image_path):
    """Mengatur background image dengan overlay gelap transparan."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        
        css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

def load_chord_diagram(chord_name):
    """Mencari path gambar diagram chord berdasarkan nama."""
    clean_name = chord_name.replace("-Chord", "") 
    possible_names = [chord_name, clean_name]
    for name in possible_names:
        for ext in (".png", ".jpg", ".jpeg"):
            path = f"chord_diagrams/{name}{ext}"
            if os.path.exists(path):
                return path
    return None

@st.cache_resource
def load_yolo_model():
    """Memuat model YOLO dari Hugging Face Hub."""
    try:
        model_path = hf_hub_download(
            repo_id="yosidanasmoro03/bestModels",
            filename="best.pt"
        )
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Load model di awal
model = load_yolo_model()

# ==========================================
# 3. VIDEO PROCESSORS (WEBRTC LOGIC)
# ==========================================

class QuizProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.target_chord = None
        self.lock = threading.Lock()
        self.correct_detected = False 

    def update_target(self, target):
        with self.lock:
            self.target_chord = target
            self.correct_detected = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Deteksi YOLO
        results = self.model(img, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()

        current_target = None
        with self.lock:
            current_target = self.target_chord

        if len(results[0].boxes.cls) > 0:
            detected_idx = int(results[0].boxes.cls[0])
            detected_name = results[0].names[detected_idx]

            # Logika Cek Jawaban
            if current_target and detected_name == current_target:
                cv2.rectangle(annotated_frame, (50, 50), (450, 150), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"BENAR: {detected_name}", (60, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                with self.lock:
                    self.correct_detected = True
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

class RealtimeProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ==========================================
# 4. FUNGSI RENDER HALAMAN (PAGE VIEWS)
# ==========================================

def next_quiz_question():
    """Mengacak soal kuis selanjutnya."""
    q_text, q_target = random.choice(list(QUIZ_DATA.items()))
    st.session_state.quiz_target = q_target
    st.session_state.quiz_text = q_text
    st.session_state["force_rerun"] = True

def render_home_page():
    """Menampilkan Halaman Utama (Home)."""
    set_background_overlay(r"backgrounds/guitar-unsplash.jpg")
    
    st.markdown('<h1 style="text-align:center; margin-bottom: 2rem;">🎸 Aplikasi Deteksi Chord Gitar</h1>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3, gap="medium")

    # Card 1: Kuis
    with c1:
        img_b64 = get_img_as_base64("kuis.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>🎸 Kuis Deteksi</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Kuis"></div>
            <p>Jawab pertanyaan kuis dengan menunjukkan chord.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("MULAI KUIS", key="home_quiz", use_container_width=True):
            st.session_state.menu = "🎸 Kuis"
            st.rerun()
    
    # Card 2: Realtime
    with c2:
        img_b64 = get_img_as_base64("realtime.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>🎥 Deteksi Live</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Live"></div>
            <p>Deteksi bebas menggunakan kamera langsung.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("BUKA KAMERA", key="home_real", use_container_width=True):
            st.session_state.menu = "🎥 Real-time"
            st.rerun()

    # Card 3: Upload
    with c3:
        img_b64 = get_img_as_base64("upload.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>📷 Upload Foto</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Upload"></div>
            <p>Upload gambar statis untuk dideteksi.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("UPLOAD GAMBAR", key="home_up", use_container_width=True):
            st.session_state.menu = "📷 Upload"
            st.rerun()

def render_quiz_page():
    """Menampilkan Halaman Kuis."""
    set_background_overlay(r"backgrounds/acoustic-guitar-dark-surroundings.jpg")
    
    st.markdown("<h3 style='text-align: center; margin:0; padding:0; color:white;'>🎸 Kuis Chord</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    # Inisialisasi soal jika belum ada
    if "quiz_target" not in st.session_state:
        next_quiz_question()
        st.rerun()

    # Layout: [Spacer, Kamera, Info, Spacer]
    c_pad_l, col_cam, col_info, c_pad_r = st.columns([0.5, 3, 2, 0.5], gap="large")

    with col_cam:
        st.write("###### Kamera")
        ctx = webrtc_streamer(
            key="quiz_compact",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=QuizProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_info:
        st.info(f"Target: **{st.session_state.quiz_target}**")
        st.markdown(f"#### {st.session_state.quiz_text}")
        
        d_path = load_chord_diagram(st.session_state.quiz_target)
        if d_path:
            st.image(d_path, caption=None, use_container_width=True) 

    # Update target ke backend processor
    if ctx.video_processor:
        ctx.video_processor.update_target(st.session_state.quiz_target)

    # Logika Loop Auto-Next
    if ctx.state.playing:
        placeholder = st.empty()
        while ctx.state.playing:
            if ctx.video_processor and ctx.video_processor.correct_detected:
                placeholder.success(f"🎉 BENAR! {st.session_state.quiz_target}")
                time.sleep(1.0)
                next_quiz_question()
                st.rerun()
                break 
            time.sleep(0.2)

def render_realtime_page():
    """Menampilkan Halaman Real-time Detection."""
    set_background_overlay(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h3 style='text-align: center; margin:0; color:white;'>🎥 Mode Real-time</h3>", unsafe_allow_html=True)
    
    # Layout Tengah
    c_pad_l, c_main, c_pad_r = st.columns([1, 2, 1])
    
    with c_main:
        st.write("###### Live Camera")
        webrtc_streamer(
            key="realtime_compact",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=RealtimeProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

def render_upload_page():
    """Menampilkan Halaman Upload Gambar."""
    set_background_overlay(r"backgrounds/adi-unsplash.jpg")
    st.markdown("<h3 style='text-align: center; margin:0; color:white;'>📷 Upload Gambar</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    
    with c1:
        uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png"])
    
    with c2:
        if uploaded_file is not None:
            bytes_data = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            
            # Prediksi
            if model:
                result = model.predict(image, verbose=False, conf=0.5)
                annotated_frame = result[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_frame_rgb, use_container_width=True)

                if len(result[0].boxes.cls) > 0:
                    detected = list(set([result[0].names[int(c)] for c in result[0].boxes.cls]))
                    st.success(f"Hasil: **{', '.join(detected)}**")
                else:
                    st.warning("Tidak terdeteksi.")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    # Load CSS Global
    load_css("styles/styles.css")
    
    # Inisialisasi State Menu
    if "menu" not in st.session_state:
        st.session_state.menu = "🏠 Home"

    # --- NAVBAR ---
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    nav_items = ["🏠 Home", "🎸 Kuis", "🎥 Real-time", "📷 Upload"]
    cols = st.columns(len(nav_items))
    
    for i, item in enumerate(nav_items):
        with cols[i]:
            if st.button(item, key=f"nav_main_{i}", use_container_width=True):
                st.session_state.menu = item
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- ROUTING HALAMAN ---
    menu = st.session_state.menu

    if menu == "🏠 Home":
        render_home_page()
    elif menu == "🎸 Kuis":
        render_quiz_page()
    elif menu == "🎥 Real-time":
        render_realtime_page()
    elif menu == "📷 Upload":
        render_upload_page()

if __name__ == "__main__":
    main()
