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
        self.target_chord = None # Contoh: "C-Chord"
        self.lock = threading.Lock()
        
    def update_target(self, target):
        with self.lock:
            self.target_chord = target

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Prediksi YOLO
        results = self.model(img, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Ambil target saat ini dari thread-safe variable
        current_target = None
        with self.lock:
            current_target = self.target_chord

        if len(results[0].boxes.cls) > 0:
            # Ambil nama chord yang terdeteksi
            detected_idx = int(results[0].boxes.cls[0])
            detected_name = results[0].names[detected_idx] # Contoh: "C-Chord"
            
            # === LOGIKA UTAMA PERUBAHAN: KOMUNIKASI DENGAN SESSION STATE ===
            if current_target and detected_name == current_target:
                # Jika Benar: Set flag di session state
                # Penting: Pastikan st.session_state diakses thread-safe
                st.session_state["chord_detected_correctly"] = True
                
                # Visualisasi BENAR di video stream
                cv2.rectangle(annotated_frame, (50, 50), (450, 150), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"✅ BENAR: {detected_name}", (60, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            else:
                # Visualisasi Deteksi Salah
                cv2.putText(annotated_frame, f"Deteksi: {detected_name}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Jika tidak ada deteksi, reset flag (agar tidak lompat terus)
            st.session_state["chord_detected_correctly"] = False
            
        
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
        if os.path.exists(path):
            return path
    return None

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
    # Pilih soal baru
    q_text, q_target = random.choice(list(questionList.items()))
    st.session_state.quiz_target = q_target
    st.session_state.quiz_text = q_text
    # Reset flag deteksi
    st.session_state["chord_detected_correctly"] = False
    st.rerun()

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

    # Inisialisasi State Pertanyaan
    if "quiz_target" not in st.session_state:
        next_quiz_question()

    # == LOGIKA LOMPAT OTOMATIS ==
    if st.session_state["chord_detected_correctly"]:
        st.success(f"Chord **{st.session_state.quiz_target}** terdeteksi dengan Benar! Melanjutkan ke soal berikutnya...")
        time.sleep(1) # Beri waktu sejenak agar user melihat pesan sukses
        next_quiz_question()
        st.stop() # Hentikan eksekusi di sini, Rerun sudah dipanggil
        
    # Layout Pertanyaan
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown(f"### Pertanyaan: \n## **{st.session_state.quiz_text}**")
        st.info(f"Target Chord: **{st.session_state.quiz_target}**")
    
    with c2:
        # Tampilkan Diagram
        diagram_path = loadChordDiagram(st.session_state.quiz_target)
        if diagram_path:
            st.image(diagram_path, caption=f"Diagram {st.session_state.quiz_target}")
        else:
            st.warning(f"Gambar diagram tidak ditemukan untuk {st.session_state.quiz_target}")

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
    if st.button("Lewati Soal", type="secondary"):
        next_quiz_question()

#==========================================
# DETEKSI REAL-TIME

elif menu == "🎥 Deteksi Real-time":
    setBackground(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h1 style='text-align: center;'>🎥 Deteksi Chord Gitar Real-time</h1>", unsafe_allow_html=True)

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



