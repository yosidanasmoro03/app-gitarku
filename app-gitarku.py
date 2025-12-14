import os
import streamlit as st
import cv2
from ultralytics import YOLO
import random
import numpy as np
import time
import base64
import av
import threading
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================

st.set_page_config(layout="wide", page_title="App Gitarku", initial_sidebar_state="collapsed")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load CSS Eksternal (Hanya ini yang dibutuhkan)
def loadCss(filePath):
    if os.path.exists(filePath):
        with open(filePath, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

loadCss("styles/styles.css") # Panggil fungsi loadCss di awal

# Set Background
def setBackground(imagePath):
    if os.path.exists(imagePath):
        with open(imagePath, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# Helper Image UI
def render_image(filename):
    if os.path.exists(filename):
        st.image(filename, use_container_width=True)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def loadModel():
    try:
        modelPath = hf_hub_download(
            repo_id="yosidanasmoro03/bestModels",
            filename="best.pt"
        )
        return YOLO(modelPath)
    except Exception:
        return None

model = loadModel()
if model is None:
    st.error("Gagal memuat model. Cek koneksi internet.")
    st.stop()

# ==========================================
# LOGIKA KUIS
# ==========================================

def next_quiz_question():
    questionList = {
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
    q_text, q_target = random.choice(list(questionList.items()))
    st.session_state.quiz_target = q_target
    st.session_state.quiz_text = q_text
    st.session_state["force_rerun"] = True

def loadChordDiagram(namaChord):
    clean_name = namaChord.replace("-Chord", "") 
    possible_names = [namaChord, clean_name]
    for name in possible_names:
        for ext in (".png", ".jpg", ".jpeg"):
            path = f"chord_diagrams/{name}{ext}"
            if os.path.exists(path):
                return path
    return None

# ==========================================
# PROCESSORS
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
        results = self.model(img, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()

        current_target = None
        with self.lock:
            current_target = self.target_chord

        if len(results[0].boxes.cls) > 0:
            detected_idx = int(results[0].boxes.cls[0])
            detected_name = results[0].names[detected_idx]

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
# NAVIGATION & HEADER
# ==========================================

if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Home"

# Navbar Renderer (Menggunakan kelas CSS 'navbar' dan 'nav-btn')
navItems = ["🏠 Home", "🎸 Kuis", "🎥 Real-time", "📷 Upload"]

st.markdown('<div class="navbar" style="text-align:center;">', unsafe_allow_html=True)
cols = st.columns(len(navItems))
for i, item in enumerate(navItems):
    with cols[i]:
        activeClass = "active" if st.session_state.menu == item else ""
        
        # Menggunakan HTML/Markdown untuk merender tombol dengan kelas CSS
        # Ini penting agar styling dari styles.css berfungsi
        buttonHtml = f"""
        <button class="nav-btn {activeClass}" onclick="window.parent.postMessage('streamlit:rerun', '*');" name="nav_click" value="{item}">
            {item}
        </button>
        """
        st.markdown(buttonHtml, unsafe_allow_html=True)
        
        # Logika Reruns Streamlit (Jika tombol diklik)
        if st.session_state.get(f"nav_click_{i}") != item and st.session_state.get(f"nav_click_{i}") is not None:
             st.session_state.menu = st.session_state[f"nav_click_{i}"]
             st.rerun()

menu = st.session_state.menu

# Logika untuk menangkap klik dari tombol custom (perlu workaround)
# Karena kita tidak bisa menggunakan st.button(), kita pakai st.form yang tersembunyi
# Namun, cara paling stabil adalah menggunakan st.button biasa dengan CSS override

# Karena kita tidak bisa menggunakan form di Streamlit, kita kembali ke st.button() 
# dan mengandalkan CSS override pada stButton > button
st.markdown('</div>', unsafe_allow_html=True) # Tutup div navbar

# Kita ulang navbar menggunakan st.button biasa, karena HTML custom tidak bisa memicu st.rerun
# Tanpa ini, navigasi tidak akan berfungsi.

# Hapus navbar HTML di atas dan ganti dengan ini untuk fungsionalitas:
st.markdown('<div class="navbar" style="margin-top: -1.5rem;">', unsafe_allow_html=True)
nav_cols = st.columns(len(navItems))
for i, item in enumerate(navItems):
    with nav_cols[i]:
        # Gunakan key unik. CSS styles.css akan menimpa tampilan tombol ini.
        if st.button(item, key=f"nav_func_{i}", use_container_width=True):
            st.session_state.menu = item
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
# Sisa elemen navbar yang tidak ter-wrap st.button akan ditangani oleh CSS di styles.css

# ==========================================
# 1. HALAMAN HOME
# ==========================================
if menu == "🏠 Home":
    setBackground(r"backgrounds/guitar-unsplash.jpg")
    
    st.markdown('<div class="title">🎸 Aplikasi Deteksi Chord Gitar</div>', unsafe_allow_html=True)
    
    # Gunakan layout yang dirancang untuk CSS card-container
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    home_cols = st.columns(3, gap="large")

    with home_cols[0]:
        st.markdown("""
        <div class="card" style="background-color: rgba(255, 255, 255, 0.9);">
            <div class="card-title">🎸 Kuis Deteksi Chord</div>
            <img src="data:image/png;base64,{}" alt="Kuis">
            <div class="card-desc">Siapkan gitar Anda. Sistem akan meminta Anda membentuk chord tertentu dan akan mendeteksinya secara otomatis untuk melanjutkan ke soal berikutnya.</div>
        </div>
        """.format(base64.b64encode(open("kuis.png", "rb").read()).decode()), unsafe_allow_html=True)
        if st.button("MULAI KUIS", key="home_quiz", use_container_width=True):
            st.session_state.menu = "🎸 Kuis"
            st.rerun()
    
    with home_cols[1]:
        st.markdown("""
        <div class="card" style="background-color: rgba(255, 255, 255, 0.9);">
            <div class="card-title">🎥 Deteksi Real-time</div>
            <img src="data:image/png;base64,{}" alt="Real-time">
            <div class="card-desc">Gunakan kamera Anda untuk mendeteksi chord gitar secara langsung tanpa batasan soal atau waktu.</div>
        </div>
        """.format(base64.b64encode(open("realtime.png", "rb").read()).decode()), unsafe_allow_html=True)
        if st.button("BUKA KAMERA", key="home_real", use_container_width=True):
            st.session_state.menu = "🎥 Real-time"
            st.rerun()

    with home_cols[2]:
        st.markdown("""
        <div class="card" style="background-color: rgba(255, 255, 255, 0.9);">
            <div class="card-title">📷 Upload Gambar</div>
            <img src="data:image/png;base64,{}" alt="Upload">
            <div class="card-desc">Unggah foto statis chord yang sedang Anda mainkan, dan sistem akan mengidentifikasi chord tersebut.</div>
        </div>
        """.format(base64.b64encode(open("upload.png", "rb").read()).decode()), unsafe_allow_html=True)
        if st.button("UPLOAD FOTO", key="home_up", use_container_width=True):
            st.session_state.menu = "📷 Upload"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 2. HALAMAN KUIS (COMPACT MODE)
# ==========================================
elif menu == "🎸 Kuis":
    setBackground(r"backgrounds/acoustic-guitar-dark-surroundings.jpg")
    
    st.markdown("<h1 style='text-align: center;'>🎸 Kuis Chord</h1>", unsafe_allow_html=True)

    if "quiz_target" not in st.session_state:
        next_quiz_question()
        st.rerun()

    col_cam, col_info = st.columns([1.5, 1], gap="small")

    with col_cam:
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
        st.markdown(f"**{st.session_state.quiz_text}**")
        
        d_path = loadChordDiagram(st.session_state.quiz_target)
        if d_path:
            st.image(d_path, caption=None, use_container_width=True)
        
        # Tombol Lewati
        if st.button("Lewati Soal ➡", key="skip_btn", use_container_width=True):
            next_quiz_question()
            st.rerun()

    if ctx.video_processor:
        ctx.video_processor.update_target(st.session_state.quiz_target)

    # Auto-Next Logic
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

# ==========================================
# 3. HALAMAN REALTIME
# ==========================================
elif menu == "🎥 Real-time":
    setBackground(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>🎥 Mode Real-time</h1>", unsafe_allow_html=True)
    
    c_pad_l, c_main, c_pad_r = st.columns([1, 4, 1])
    with c_main:
        webrtc_streamer(
            key="realtime_compact",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=RealtimeProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# ==========================================
# 4. HALAMAN UPLOAD
# ==========================================
elif menu == "📷 Upload":
    setBackground(r"backgrounds/adi-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>📷 Upload Gambar</h1>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    
    with c1:
        uploadedFile = st.file_uploader("Pilih gambar", type=["jpg", "png"])

    with c2:
        if uploadedFile is not None:
            bytesData = np.frombuffer(uploadedFile.read(), np.uint8)
            image = cv2.imdecode(bytesData, cv2.IMREAD_COLOR)
            
            result = model.predict(image, verbose=False, conf=0.5)
            annotatedFrame = result[0].plot()
            annotatedFrameRGB = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB)
            
            st.image(annotatedFrameRGB, use_container_width=True)

            if len(result[0].boxes.cls) > 0:
                detected = list(set([result[0].names[int(c)] for c in result[0].boxes.cls]))
                st.success(f"Hasil: **{', '.join(detected)}**")
            else:
                st.warning("Tidak terdeteksi.")
