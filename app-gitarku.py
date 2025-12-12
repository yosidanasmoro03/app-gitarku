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

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#=================================
# Fungsi untuk load CSS eksternal

def loadCss(filePath):
    with open(filePath, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#=====================================
# Fungsi untuk background dan overlay

def setBackground(imagePath):
    with open(imagePath, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

#====================================
# Load model YOLO

@st.cache_resource
def loadModel():
    modelPath = hf_hub_download(
        repo_id="yosidanasmoro03/bestModels",
        filename="best.pt"
    )
    return YOLO(modelPath)

model = loadModel()

# ===============================
# KELAS PEMROSES VIDEO
# ===============================

# Kelas untuk Kuis
class QuizProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.target_chord = None
        self.lock = threading.Lock()
        self.detected_chord = None
        self.is_correct = False

    def update_target(self, target):
        with self.lock:
            self.target_chord = target
            self.is_correct = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Prediksi YOLO
        results = self.model(img, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Logika Pengecekan Kuis
        current_target = None
        with self.lock:
            current_target = self.target_chord

        if len(results[0].boxes.cls) > 0:
            # Ambil nama chord yang terdeteksi pertama
            detected_idx = int(results[0].boxes.cls[0])
            detected_name = results[0].names[detected_idx]
            
            # Simpan state (opsional)
            with self.lock:
                self.detected_chord = detected_name

            # Validasi Jawaban Visual di Layar
            if current_target and detected_name == current_target:
                # Jika Benar: Gambar kotak HIJAU dan teks BENAR
                cv2.rectangle(annotated_frame, (50, 50), (300, 150), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"BENAR: {detected_name}!", (60, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                with self.lock:
                    self.is_correct = True
            else:
                # Jika Salah: Tampilkan apa yang terdeteksi
                cv2.putText(annotated_frame, f"Deteksi: {detected_name}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Kleas untuk realtime 
class RealtimeProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

#=======================================
# fungsi untuk menampilkan gambar chord di mode kuis
 
def loadChordDiagram(namaChord):
    for ext in (".png", ".jpg", ".jpeg"):
        path = f"chord_diagrams/{namaChord}{ext}"
        if os.path.isfile(path):
            return path
    return None

#====================================
# Navbar dan Menu

loadCss("styles/styles.css")

if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Home"

navItems = ["🏠 Home", "🎸 Kuis Deteksi Chord", "🎥 Deteksi Real-time", "📷 Upload Gambar"]

#Render navbar
st.markdown('<div class="navbar">', unsafe_allow_html=True)
cols = st.columns(len(navItems))

for i, item in enumerate(navItems):
    with cols[i]:
        activeClass = "active" if st.session_state.menu == item else ""
        buttonHtml = f"""
        <form action="" method="get" style="margin:0; padding:0;">
            <button name="nav" value="{item}" class="nav-btn {activeClass}">{item}</button>
        </form>
        """

        st.markdown(buttonHtml, unsafe_allow_html=True)

        queryParams = st.query_params
        if "nav" in queryParams and queryParams["nav"] == item:
            st.session_state.menu = item
            st.query_params.clear()
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
menu = st.session_state.menu

#==================================
# HALAMAN HOME

if menu == "🏠 Home":
    setBackground(r"backgrounds/guitar-unsplash.jpg")

    st.markdown('<div class="title">🎸 Aplikasi Deteksi Chord Gitar</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.info("🎸 **Kuis Deteksi**")
        st.write("Siapkan gitar dan jawab pertanyaan dengan menunjukkan chord yang diminta di depan kamera.")
        if st.button("MULAI KUIS", key="quiz_home"):
            st.session_state.menu = "🎸 Kuis Deteksi Chord"
            st.rerun()
    
    with col2:
        st.warning("🎥 **Real-time**")
        st.write("Gunakan kamera untuk mendeteksi chord secara langsung.")
        if st.button("BUKA KAMERA", key="realtime_home"):
            st.session_state.menu = "🎥 Deteksi Real-time"
            st.rerun()

    with col3:
        st.success("📷 **Upload**")
        st.write("Upload gambar untuk mendeteksi chord yang dimainkan.")
        if st.button("UPLOAD GAMBAR", key="upload_home"):
            st.session_state.menu = "📷 Upload Gambar"
            st.rerun()

#======================================
# HALAMAN KUIS DETEKSI CHORD

elif menu == "🎸 Kuis Deteksi Chord":
    setBackground(r"backgrounds/acoustic-guitar-dark-surroundings.jpg")
    st.markdown("<h1 style='text-align: center;'>🎸 Kuis Chord Gitar dengan Kamera</h1>", unsafe_allow_html=True)

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

    # Inisialisasi State Pertanyaan
    if "quiz_target" not in st.session_state:
        # Pilih random pertama kali
        key, val = random.choice(list(questionList.items()))
        st.session_state.quiz_target = key # Key dict (Misal "C") adalah label class YOLO
        st.session_state.quiz_text = val

    # Layout Pertanyaan
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown(f"### Pertanyaan: \n## **{st.session_state.quiz_text}**")
        st.info(f"Target Chord: **{st.session_state.quiz_target}**")
    
    with c2:
        # Tampilkan Diagram
        diagram_path = loadChordDiagram(st.session_state.quiz_target)
        st.image(diagram_path, caption=f"Diagram {st.session_state.quiz_target}")

    st.write("---")
    
    # WEBRTC STREAMER
    # Kita menggunakan `ctx` untuk mengakses processor
    ctx = webrtc_streamer(
        key="quiz_detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=QuizProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Kirim Target Chord ke Processor agar bisa dicek real-time
    if ctx.video_processor:
        ctx.video_processor.update_target(st.session_state.quiz_target)

    # Tombol Ganti Soal
    st.write("")
    if st.button("➡ Soal Selanjutnya (Acak)", type="primary"):
        key, val = random.choice(list(questionList.items()))
        st.session_state.quiz_target = key
        st.session_state.quiz_text = val
        st.rerun()

#==========================================
# DETEKSI REAL-TIME

elif menu == "🎥 Deteksi Real-time":
    setBackground(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>🎥 Deteksi Chord Gitar Real-time</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;'>Streaming kamera browser, YOLO berjalan secara real-time.</p>",
        unsafe_allow_html=True
    )

    st.write("Pastikan memberikan izin akses kamera pada browser.")

    webrtc_streamer(
        key="realtime_detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=RealtimeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

#=========================================
# DETEKSI GAMBAR UPLOAD

elif menu == "📷 Upload Gambar":
    setBackground(r"backgrounds/adi-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>📷 Deteksi Chord dari Gambar</h1>", unsafe_allow_html=True)

    uploadedFile = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploadedFile is not None:
        bytesData = np.frombuffer(uploadedFile.read(), np.uint8)
        image = cv2.imdecode(bytesData, cv2.IMREAD_COLOR)
        result = model.predict(image, verbose=False)
        annotatedFrame = result[0].plot()
        annotatedFrame = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB)
        st.image(annotatedFrame, channels="RGB", use_column_width=True)

        if len(result[0].boxes.cls) > 0:
            detected = [result[0].names[int(c)] for c in result[0].boxes.cls]
            st.success(f"Chord terdeteksi: {', '.join(detected)}")
        else:
            st.warning("Tidak ada chord terdeteksi pada gambar ini.")
