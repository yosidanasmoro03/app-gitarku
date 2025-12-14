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
# 1. KONFIGURASI HALAMAN & CSS (PENTING UNTUK UKURAN UI)
# ==========================================

st.set_page_config(layout="wide", page_title="App Gitarku", initial_sidebar_state="collapsed")

# Inject CSS untuk memangkas ruang kosong (Padding) dan mengecilkan elemen
st.markdown("""
    <style>
        /* 1. Mengurangi Padding Halaman Utama secara drastis */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100%;
        }
        
        /* 2. Mengecilkan Judul H1 */
        h1 {
            font-size: 1.8rem !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* 3. Mengatur Navbar agar lebih tipis */
        div.stButton > button {
            padding: 0.25rem 1rem !important;
            font-size: 0.9rem !important;
        }
        
        /* 4. Memastikan Video WebRTC tidak terlalu tinggi pada layar lebar */
        /* Kita batasi max-height elemen video agar tidak memaksa scroll */
        video {
            max-height: 60vh !important; 
            width: 100% !important;
            object-fit: contain !important;
        }
        
        /* 5. Membatasi tinggi gambar diagram agar proporsional */
        [data-testid="stImage"] img {
            max-height: 250px !important;
            object-fit: contain !important;
        }
    </style>
""", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def loadModel():
    # Pastikan repo_id dan filename benar
    try:
        modelPath = hf_hub_download(
            repo_id="yosidanasmoro03/bestModels",
            filename="best.pt"
        )
        return YOLO(modelPath)
    except Exception as e:
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
        "Bentuk jari chord C": "C-Chord", # Teks diperpendek agar muat di UI compact
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

# Helper Diagram
def loadChordDiagram(namaChord):
    clean_name = namaChord.replace("-Chord", "") 
    possible_names = [namaChord, clean_name]
    for name in possible_names:
        for ext in (".png", ".jpg", ".jpeg"):
            path = f"chord_diagrams/{name}{ext}"
            if os.path.exists(path):
                return path
    return None

# Helper Background
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

# Navbar Mini
navItems = ["🏠 Home", "🎸 Kuis", "🎥 Real-time", "📷 Upload"]
cols = st.columns(len(navItems))
for i, item in enumerate(navItems):
    with cols[i]:
        # Gunakan tombol kecil
        if st.button(item, key=f"nav_{i}", use_container_width=True):
            st.session_state.menu = item
            st.rerun()

menu = st.session_state.menu

# ==========================================
# 1. HALAMAN HOME
# ==========================================
if menu == "🏠 Home":
    setBackground(r"backgrounds/guitar-unsplash.jpg")
    
    st.markdown('<h1 style="text-align:center; color: white; text-shadow: 2px 2px 4px #000000;">🎸 Deteksi Chord Gitar</h1>', unsafe_allow_html=True)
    
    # Gunakan Container agar layout lebih rapi
    with st.container():
        c1, c2, c3 = st.columns(3, gap="small")

        with c1:
            st.markdown("##### 🎸 Kuis")
            render_image("kuis.png") 
            if st.button("MULAI", key="home_quiz", use_container_width=True):
                st.session_state.menu = "🎸 Kuis"
                st.rerun()
        
        with c2:
            st.markdown("##### 🎥 Live")
            render_image("realtime.png")
            if st.button("KAMERA", key="home_real", use_container_width=True):
                st.session_state.menu = "🎥 Real-time"
                st.rerun()

        with c3:
            st.markdown("##### 📷 Foto")
            render_image("upload.png")
            if st.button("UPLOAD", key="home_up", use_container_width=True):
                st.session_state.menu = "📷 Upload"
                st.rerun()

# ==========================================
# 2. HALAMAN KUIS (COMPACT MODE)
# ==========================================
elif menu == "🎸 Kuis":
    setBackground(r"backgrounds/acoustic-guitar-dark-surroundings.jpg")
    
    # Judul Kecil di Kiri atas atau Tengah
    st.markdown("<h3 style='text-align: center; margin:0; padding:0; color:white;'>🎸 Kuis Chord</h3>", unsafe_allow_html=True)

    if "quiz_target" not in st.session_state:
        next_quiz_question()
        st.rerun()

    # Layout: Kamera (Kiri - Lebar) | Info (Kanan - Sempit)
    # Rasio 1.5 : 1 agar kamera tidak terlalu tinggi
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
        # Tampilkan Info dengan padat
        st.info(f"Target: **{st.session_state.quiz_target}**")
        st.markdown(f"**{st.session_state.quiz_text}**")
        
        d_path = loadChordDiagram(st.session_state.quiz_target)
        if d_path:
            # Gambar diagram dibatasi tingginya oleh CSS
            st.image(d_path, caption=None, use_container_width=True)
        
        if st.button("Lewati ➡", key="skip_btn", use_container_width=True):
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
    st.markdown("<h3 style='text-align: center; margin:0; color:white;'>🎥 Mode Real-time</h3>", unsafe_allow_html=True)
    
    # Gunakan kolom tengah agar video tidak terlalu lebar (dan tinggi)
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
    st.markdown("<h3 style='text-align: center; margin:0; color:white;'>📷 Upload Gambar</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    
    with c1:
        uploadedFile = st.file_uploader("Pilih gambar", type=["jpg", "png"])

    with c2:
        if uploadedFile is not None:
            bytesData = np.frombuffer(uploadedFile.read(), np.uint8)
            image = cv2.imdecode(bytesData, cv2.IMREAD_COLOR)
            
            # Resize image agar processing lebih cepat dan tampilan tidak raksasa
            # Opsional: image = cv2.resize(image, (640, 640)) 
            
            result = model.predict(image, verbose=False, conf=0.5)
            annotatedFrame = result[0].plot()
            annotatedFrameRGB = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB)
            
            st.image(annotatedFrameRGB, use_container_width=True)

            if len(result[0].boxes.cls) > 0:
                detected = list(set([result[0].names[int(c)] for c in result[0].boxes.cls]))
                st.success(f"Hasil: **{', '.join(detected)}**")
            else:
                st.warning("Tidak terdeteksi.")
