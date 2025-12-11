import os
import streamlit as st
import cv2
from ultralytics import YOLO
import random
import numpy as np
import time
import base64
import torch
import requests

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
def loadModel(url, outPath = "models/best.pt"):
    if not os.path.exists(outPath):
        r = requests.get(url)
        with open (outPath, "wb") as f:
            f.write(r.content)
    return YOLO(outPath)

model = loadModel("https://huggingface.co/yosidanasmoro03/bestModels/resolve/main/models/best.pt")

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
        st.image("kuis.png", use_container_width=True)
        st.markdown('<div class="card-title">Kuis Deteksi Chord</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Siapkan gitar dan jawab pertanyaan dengan menunjukkan chord yang diminta di depan kamera.</div>', unsafe_allow_html=True)

        if st.button("MULAI KUIS", key="quiz"):
            st.session_state.menu = "🎸 Kuis Deteksi Chord"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("realtime.png", use_container_width=True)
        st.markdown('<div class="card-title">Deteksi Real-time</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Gunakan kamera untuk mendeteksi chord secara langsung.</div>', unsafe_allow_html=True)

        if st.button("BUKA KAMERA", key="realtime"):
            st.session_state.menu = "🎥 Deteksi Real-time"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("upload.png", use_container_width=True)
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

    # Membuat session state
    for key in ["isActive", "currentQuestion", "expectedChord"]:
        if key not in st.session_state:
            st.session_state[key] = "" if key != "isActive" else False
    
    #tombol kontrol kuis
    col1, col2 = st.columns(2)
    if col1.button("MULAI KUIS", key="startQuiz"):
        st.session_state.isActive = True
        question, chord = random.choice(list(questionList.items()))
        st.session_state.currentQuestion = question
        st.session_state.expectedChord = chord
    
    if col2.button("❌ STOP KUIS", key="stopQuiz"):
        st.session_state.isActive = False
        st.success("Kuis dihentikan")
        st.stop()
    
    #jalankan kuis
    if st.session_state.isActive:
        st.subheader("Pertanyaan:")
        st.markdown(f"**{st.session_state.currentQuestion}**")

        camera = cv2.VideoCapture(0)
        # layout 2 kolom: kiri kamera, kanan diagram chord
        colCam, colDiag = st.columns([3,1])

        frameContainer = colCam.empty()
        messageContainer = colCam.empty()

        #menampilkan diagram chord sesuai soal
        diagramPath = loadChordDiagram(st.session_state.expectedChord)

        with colDiag:
            st.subheader("Bentuk Chord:")
            st.image(diagramPath, use_container_width=True)
        
        while st.session_state.isActive and camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                st.error("Tidak dapat mengakses kamera")
                break
            
            result = model.predict(frame, verbose=False)
            label = result[0].boxes.cls.cpu().numpy()
            annotatedFrame = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)

            #tampilkan video
            frameContainer.image(annotatedFrame, channels="RGB", use_container_width=True)

            if len(label) > 0:
                detectedChord = result[0].names[int(label[0])]
                expectedChord = st.session_state.expectedChord

                if detectedChord == expectedChord:
                    messageContainer.success(f"✅ Benar! Chord {detectedChord} terdeteksi")
                    time.sleep(2)

                    #update soal & chord berikutnya
                    question, chord = random.choice(list(questionList.items()))
                    st.session_state.currentQuestion = question
                    st.session_state.expectedChord = chord

                    st.rerun()
                
                else:
                    if detectedChord != expectedChord:
                        messageContainer.warning(f"Terdeteksi: {detectedChord}. Coba lagi!")
                
            else:
                messageContainer.info("Menunggu deteksi...")

            time.sleep(0.1)
            
        camera.release()

#==========================================
# DETEKSI REAL-TIME

elif menu == "🎥 Deteksi Real-time":
    setBackground(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>🎥 Deteksi Chord Gitar Real-time</h1>", unsafe_allow_html=True)

    if "cameraActive" not in st.session_state:
        st.session_state.cameraActive = False
    
    col1, col2 = st.columns(2)
    if col1.button("MULAI DETEKSI", key="startRealtime"):
        st.session_state.cameraActive = True
    if col2.button("❌ MATIKAN KAMERA", key="stopRealtime"):
        st.session_state.cameraActive = False
    
    framePlaceholder = st.empty()

    if st.session_state.cameraActive:
        camera = cv2.VideoCapture(0)
        while st.session_state.cameraActive:
            ret, frame = camera.read()
            if not ret:
                st.error("Kamera tidak dapat diakses.")
                break
            result = model.predict(frame, verbose=False)
            annotatedFrame = result[0].plot()
            annotatedFrame = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2RGB)
            framePlaceholder.image(annotatedFrame, channels="RGB", use_container_width=True)
            time.sleep(0.1)
            if not st.session_state.cameraActive:
                break

        camera.release()
    st.success("Kamera dimatikan.")

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
        st.image(annotatedFrame, channels="RGB", use_container_width=True)

        if len(result[0].boxes.cls) > 0:
            detected = [result[0].names[int(c)] for c in result[0].boxes.cls]
            st.success(f"Chord terdeteksi: {', '.join(detected)}")
        else:
            st.warning("Tidak ada chord terdeteksi pada gambar ini.")





