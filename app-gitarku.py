import os
import numpy as np
import random
import cv2
import av
import time
import base64
import threading

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Konfigurasi
#============================================

st.set_page_config(
    layout="wide",
    page_title="App Gitarku"
    initial_sidebar_state="collapsed"
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}]}
)

# daftar pertanyaan kuis
QUIZ_DATA = {
    "Bentuk jari chord C": "C-Chord",
    "Tunjukkan chord G dengan tanganmu": "G-Chord",
    "Bentuk chord D di gitar": "D-Chord",
    "Bentuk chord A di fretboard": "A-Chord",
    "Tunjukkan chord E": "E-Chord",
    "Tunjukkan posisi jari untuk chord Am": "Am-Chord",
    "Tampilkan chord F dengan benar": "F-Chord",
    "Bentuk chord Bm di gitar": "Bm-Chord",
    "Tunjukkan chord Dm": "Dm-Chord",
    "Cobalah bentuk chord Em": "Em-Chord"
}

# Fungsi utility (css, gambar, model)
#=============================================

def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_img_as_base64(file_path):
    """mengubah file gambar jadi string base64 untuk injection HTML"""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_overlay(image_path):
    """background menjadi gelap transparan"""
    with open (image_path, "rb") as f:
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
            return path
    return None

@st.cache_resource
def load_yolo_model():
    try:
        model_path = hf_hub_download(
            repo_id="yosidanasmoro03/bestModels",
            filename="best.pt"
        )
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# load model di awal
model = load_yolo_model()

# Logic WebRTC (kamera)
#===================================================

class QuizProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.target_chord = None
        self.lock = threading.Lock()
        self.correct_detected = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        #Deteksi YOLO
        results = self.model(img, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()

        current_target = None
        with self.lock:
            current_target = self.target_chord

        if len(results[0].boxes.cls) > 0:
            detected_idx = int(results[0].boxes.cls[0])
            detected_name = results[0].names[detected_idx]

            #logic cek jawaban
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

# Fungsi render halaman
#===================================================

def next_quiz_question():
    """mengacak soal kuis selanjutnya"""
    q_text, q_target = random.choice(list(QUIZ_DATA.items()))
    st.session_state.quiz_target = q_target
    st.session_state.quiz_text = q_text
    st.session_state["force_rerun"] = True

#halaman utama
def render_home_page():
    set_background_overlay(r"backgrounds/guitar-unsplash.jpg")

    st.markdown('<h1 style="text-align:center; margin-bottom: 2rem;">🎸 Aplikasi Deteksi Chord Gitar</h1>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3, gap="medium")

    # card 1: kuis
    with c1:
        img_b64 = get_img_as_base64("kuis.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>🎸 Kuis Deteksi Chord</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Kuis"></div>
            <p>Siapkan gitar dan jawab pertanyaan dengan menunjukkan chord yang diminta di depan kamera.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("MULAI KUIS", key="home_quiz", use_container_width=True):
            st.session_state.menu = "🎸 Kuis"
            st.rerun()

    # card 2: realtime
    with c2:
        img_b64 = get_img_as_base64("realtime.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>🎥 Deteksi Real-time</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Live"></div>
            <p>Deteksi chord bebas menggunakan kamera langsung.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("BUKA KAMERA", key="home_real", use_container_width=True):
            st.session_state.menu = "🎥 Real-time"
            st.rerun()

    # card 3: upload gambar
    with c3:
        img_b64 = get_img_as_base64("upload.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>📷 Upload Gambar</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Upload"></div>
            <p>Upload gambar untuk mendeteksi chord yang dimainkan.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("UPLOAD GAMBAR", key="home_up", use_container_width=True):
            st.session_state.menu = "📷 Upload"
            st.rerun()

#halaman fitur kuis
def render_quiz_page():
    set_background_overlay(r"backgrounds/acoustic-guitar-dark-surroundings.jpg")

    st.markdown("<h3 style='text-align: center; margin:0; padding:0; color:white;'>🎸 Kuis Deteksi Chord</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow html=True)

    #inisialisasi soal jika belum ada
    if "quiz_target" not in st.session_state:
        next_quiz_station()
        st.rerun()

    #layout: [spacer, kamera, info, spacer]
    c_pad_1, col_cam, col_info, c_pad_r = st.columns([0.5, 3, 2, 0.5], gap="large")

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

    #update target ke backend processor
    if ctx. video_processor:
        ctx.video_processor.update_target(st.session_state.quiz_target)

    #logic loop auto-next soal
    if ctx.state.playing:
        placeholder = st.empty()
        while ctx.state.playing:
            if ctx.video_processor and ctx.video_processor.correct_detected:
                placeholder.success(f"✅ BENAR! {st.session_state.quiz_target}")
                time.sleep(1.0)
                next_quiz_station()
                st.rerun()
                break
            time.sleep(0.2)

#halaman fitur deteksi realtime
def render_realtime_page():
    set_background_overlay(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h3 style='text-align: center; margin:0; color:white;'>🎥 Deteksi Real-time</h3>", unsafe_allow_html=True)

    #layout tengah
    c_pad_1, c_main, c_pad_r = st.columns([1,2,1])

    with c_main:
        st.write("###### Kamera")
        webrtc_streamer(
            key="realtime_compact",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=RealtimeProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

#halaman upload gambar
def render_upload_page():
    set_background_overlay(r"backgrounds/adi-unsplash.jpg")
    st.markdown("<h3 style= 'text-align: center; margin:0; color:white'>📷 Upload Gambar</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])

    with c1:
        uploaded_file is not None:
            bytes_data = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)

            #prediksi
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

# MAIN EXECUTION
#===================================================

def main():
    #load css eksternal
    load_css("styles/styles.css")

    #inisialisasi state menu
    if "menu" not in st.session_state:
        st.session_state.menu = "🏠 Home"

    # NAVBAR
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    nav_items = ["🏠 Home", "🎸 Kuis", "🎥 Real-time", "📷 Upload"]
    cols = st.columns(len(nav_items))

    for i, item in enumerate(nav_items):
        with cols[i]:
            if st.button(item, key=f"nav_main_{i}", use_container_width=True):
                st.button_state.menu = item
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    #routing halaman
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
      
