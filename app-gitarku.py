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
# KONFIGURASI AWAL
# ==========================================

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load CSS
def loadCss(filePath):
    if os.path.exists(filePath):
        with open(filePath, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# Helper untuk gambar UI (Home)
def render_image(filename):
    if os.path.exists(filename):
        st.image(filename, use_container_width=True)
    else:
        st.warning(f"File {filename} tidak ditemukan.")

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def loadModel():
    modelPath = hf_hub_download(
        repo_id="yosidanasmoro03/bestModels",
        filename="best.pt"
    )
    return YOLO(modelPath)

try:
    model = loadModel()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# LOGIKA KUIS & STATE MANAGEMENT
# ==========================================

def next_quiz_question():
    questionList = {
        "Tunjukkan bentuk jari untuk chord C": "C-Chord",
        "Tunjukkan chord G dengan tanganmu": "G-Chord",
        "Sekarang, bentuk chord D di gitar": "D-Chord",
        "Bentuk chord A di fretboard": "A-Chord",
        "Coba tunjukkan chord E sekarang": "E-Chord",
        "Tunjukkan posisi jari untuk chord Am": "Am-Chord",
        "Tampilkan chord F dengan benar": "F-Chord",
        "Bentuk chord Bm di gitar": "Bm-Chord",
        "Ayo tunjukkan chord Dm dengan posisi tangan yang benar": "Dm-Chord",
        "Cobalah bentuk chord Em": "Em-Chord"
    }
    q_text, q_target = random.choice(list(questionList.items()))
    st.session_state.quiz_target = q_target
    st.session_state.quiz_text = q_text
    st.session_state["force_rerun"] = True # Trigger rerun

# ==========================================
# VIDEO PROCESSORS
# ==========================================

class QuizProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.target_chord = None
        self.lock = threading.Lock()
        # Flag untuk komunikasi ke UI
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

            # Cek jawaban
            if current_target and detected_name == current_target:
                # Visualisasi BENAR (Hijau)
                cv2.rectangle(annotated_frame, (50, 50), (450, 150), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"BENAR: {detected_name}!", (60, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                
                # Set flag agar UI tahu
                with self.lock:
                    self.correct_detected = True
            else:
                # Visualisasi SALAH/Deteksi (Merah)
                cv2.putText(annotated_frame, f"Deteksi: {detected_name}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

class RealtimeProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Helper Diagram
def loadChordDiagram(namaChord):
    # Support "C-Chord" atau "C"
    clean_name = namaChord.replace("-Chord", "") 
    possible_names = [namaChord, clean_name]
    for name in possible_names:
        for ext in (".png", ".jpg", ".jpeg"):
            path = f"chord_diagrams/{name}{ext}"
            if os.path.exists(path):
                return path
    return None

# ==========================================
# MAIN APP & UI
# ==========================================

loadCss("styles/styles.css")

if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Home"

# --- NAVBAR ---
navItems = ["🏠 Home", "🎸 Kuis Deteksi Chord", "🎥 Deteksi Real-time", "📷 Upload Gambar"]
st.markdown('<div class="navbar" style="text-align:center;">', unsafe_allow_html=True)
cols = st.columns(len(navItems))
for i, item in enumerate(navItems):
    with cols[i]:
        if st.button(item, key=f"nav_{i}", use_container_width=True):
            st.session_state.menu = item
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

menu = st.session_state.menu

# ==========================================
# 1. HALAMAN HOME (Updated dengan Gambar)
# ==========================================
if menu == "🏠 Home":
    setBackground(r"backgrounds/guitar-unsplash.jpg")
    st.markdown('<div class="title" style="text-align:center;">🎸 Aplikasi Deteksi Chord Gitar</div>', unsafe_allow_html=True)
    st.write("---")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("### 🎸 Kuis Deteksi")
        # MENAMPILKAN GAMBAR kuis.png
        render_image("kuis.png") 
        st.write("Jawab pertanyaan kuis dengan menunjukkan chord.")
        if st.button("MULAI KUIS", key="quiz_home", use_container_width=True):
            st.session_state.menu = "🎸 Kuis Deteksi Chord"
            st.rerun()
    
    with col2:
        st.markdown("### 🎥 Real-time")
        # MENAMPILKAN GAMBAR realtime.png
        render_image("realtime.png")
        st.write("Deteksi bebas menggunakan kamera langsung.")
        if st.button("BUKA KAMERA", key="realtime_home", use_container_width=True):
            st.session_state.menu = "🎥 Deteksi Real-time"
            st.rerun()

    with col3:
        st.markdown("### 📷 Upload")
        # MENAMPILKAN GAMBAR upload.png
        render_image("upload.png")
        st.write("Upload gambar statis untuk dideteksi.")
        if st.button("UPLOAD GAMBAR", key="upload_home", use_container_width=True):
            st.session_state.menu = "📷 Upload Gambar"
            st.rerun()

# ==========================================
# 2. HALAMAN KUIS (Updated: Auto Next)
# ==========================================
elif menu == "🎸 Kuis Deteksi Chord":
    setBackground(r"backgrounds/acoustic-guitar-dark-surroundings.jpg")
    st.markdown("<h1 style='text-align: center;'>🎸 Kuis Chord Gitar</h1>", unsafe_allow_html=True)

    # Inisialisasi Soal Pertama
    if "quiz_target" not in st.session_state:
        next_quiz_question()
        st.rerun()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### Pertanyaan: \n## **{st.session_state.quiz_text}**")
        st.info(f"Target: **{st.session_state.quiz_target}**")
    with c2:
        d_path = loadChordDiagram(st.session_state.quiz_target)
        if d_path:
            st.image(d_path, caption=f"Diagram {st.session_state.quiz_target}")
    
    st.write("---")

    # WEBRTC STREAMER
    ctx = webrtc_streamer(
        key="quiz_detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=QuizProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Kirim target ke processor
    if ctx.video_processor:
        ctx.video_processor.update_target(st.session_state.quiz_target)

    # ============================================================
    # TRIK AUTO-NEXT: LOOPING DI UI UNTUK CEK STATUS PROCESSOR
    # ============================================================
    if ctx.state.playing:
        placeholder = st.empty()
        while ctx.state.playing:
            # Cek apakah processor mendeteksi "correct_detected"
            if ctx.video_processor and ctx.video_processor.correct_detected:
                # Tampilkan pesan sukses sebentar
                placeholder.success(f"✅ BENAR! {st.session_state.quiz_target} Terdeteksi!")
                time.sleep(1.5) # Jeda agar user lihat notifikasi
                
                # Ganti Soal & Rerun
                next_quiz_question()
                st.rerun()
                break # Keluar dari loop

            time.sleep(0.2) # Sleep kecil agar CPU tidak jebol

# ==========================================
# 3. HALAMAN REALTIME
# ==========================================
elif menu == "🎥 Deteksi Real-time":
    setBackground(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>🎥 Deteksi Real-time</h1>", unsafe_allow_html=True)
    
    webrtc_streamer(
        key="realtime_detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=RealtimeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ==========================================
# 4. HALAMAN UPLOAD
# ==========================================
elif menu == "📷 Upload Gambar":
    setBackground(r"backgrounds/adi-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>📷 Upload Gambar</h1>", unsafe_allow_html=True)

    uploadedFile = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploadedFile is not None:
        bytesData = np.frombuffer(uploadedFile.read(), np.uint8)
        image = cv2.imdecode(bytesData, cv2.IMREAD_COLOR)
        
        result = model.predict(image, verbose=False, conf=0.5)
        annotatedFrame = result[0].plot()
        annotatedFrameRGB = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB)
        
        st.image(annotatedFrameRGB, use_container_width=True)

        if len(result[0].boxes.cls) > 0:
            detected = list(set([result[0].names[int(c)] for c in result[0].boxes.cls]))
            st.success(f"Chord terdeteksi: {', '.join(detected)}")
        else:
            st.warning("Tidak ada chord terdeteksi.")

