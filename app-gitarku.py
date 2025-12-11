import os
import streamlit as st
import cv2
from ultralytics import YOLO
import random
import numpy as np
import time
import base64
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

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
        filename="models/best.pt"
    )
    return YOLO(modelPath)

model = loadModel()

class quizDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

    result = model.predict(img, verbose=False)
    annotated = result[0].plot()

    # menyimpan hasil deteksi ke session state
    if len(result[0].boxes.cls) > 0:
        detected = result[0].names[int(result[0].boxes.cls[0])]
        st.session_state.last_detected_chord = detected
    else:
        st.session_state.last_detected_chord = None
    
    return annotated

class realTimeDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = model.predict(img, verbose=False)
        annotated = result[0].plot()
        return annotated


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
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("kuis.png", use_column_width=True)
        st.markdown('<div class="card-title">Kuis Deteksi Chord</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Siapkan gitar dan jawab pertanyaan dengan menunjukkan chord yang diminta di depan kamera.</div>', unsafe_allow_html=True)

        if st.button("MULAI KUIS", key="quiz"):
            st.session_state.menu = "🎸 Kuis Deteksi Chord"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("realtime.png", use_column_width=True)
        st.markdown('<div class="card-title">Deteksi Real-time</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Gunakan kamera untuk mendeteksi chord secara langsung.</div>', unsafe_allow_html=True)

        if st.button("BUKA KAMERA", key="realtime"):
            st.session_state.menu = "🎥 Deteksi Real-time"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("upload.png", use_column_width=True)
        st.markdown('<div class="card-title">Upload Gambar</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Upload gambar gitar untuk mendeteksi chord yang dimainkan.</div>', unsafe_allow_html=True)

        if st.button("UPLOAD GAMBAR", key="upload"):
            st.session_state.menu = "📷 Upload Gambar"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

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

    # initial state
    if "isActive" not in st.session_state:
        st.session_state.isActive = False
    if "currentQuestion" not in st.session_state:
        st.session_state.currentQuestion = ""
    if "expectedChord" not in st.session_state:
        st.session_state.expectedChord = ""

    col1, col2 = st.columns(2)
    if col1.button("MULAI KUIS"):
        st.session_state.isActive = True
        q, c = random.choice(list(questionList.items()))
        st.session_state.currentQuestion = q
        st.session_state.expectedChord = c
        st.session_state.last_detected_chord = None

    if col2.button("❌ STOP KUIS"):
        st.session_state.isActive = False
        st.success("Kuis dihentikan")

    if st.session_state.isActive:
        st.subheader("Pertanyaan:")
        st.markdown(f"**{st.session_state.currentQuestion}**")

        colCam, colDiag = st.columns([3,1])

        with colDiag:
            st.subheader("Bentuk Chord yang diminta:")
            st.image(loadChordDiagram(st.session_state.expectedChord), use_column_width=True)

        # Start WebRTC streaming for quiz
        webrtc_streamer(
            key="quiz_detector",
            video_transformer_factory=QuizDetector,
            media_stream_constraints={"video": True, "audio": False},
        )

        # cek apakah chord terdeteksi
        detected = st.session_state.last_detected_chord

        if detected is not None:
            if detected == st.session_state.expectedChord:
                st.success(f"✅ Benar! Chord {detected} terdeteksi")

                time.sleep(1)

                # next question
                q, c = random.choice(list(questionList.items()))
                st.session_state.currentQuestion = q
                st.session_state.expectedChord = c
                st.session_state.last_detected_chord = None
                st.rerun()
            else:
                st.warning(f"Terdeteksi: {detected}. Coba lagi!")
        else:
            st.info("Menunggu deteksi...")

            time.sleep(0.1)
            
        camera.release()

#==========================================
# DETEKSI REAL-TIME

elif menu == "🎥 Deteksi Real-time":
    setBackground(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>🎥 Deteksi Chord Gitar Real-time</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;'>Streaming kamera browser, YOLO berjalan secara real-time.</p>",
        unsafe_allow_html=True
    )

    # STATE untuk start/stop kamera
    if "realtime_active" not in st.session_state:
        st.session_state.realtime_active = False

    col1, col2 = st.columns(2)
    if col1.button("MULAI DETEKSI"):
        st.session_state.realtime_active = True

    if col2.button("❌ MATIKAN KAMERA"):
        st.session_state.realtime_active = False
        st.experimental_rerun()

    # jika aktif, tampilkan WebRTC
    if st.session_state.realtime_active:
        webrtc_streamer(
            key="realtime_detector",
            video_transformer_factory=RealtimeDetector,
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.info("Kamera tidak aktif.")

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










