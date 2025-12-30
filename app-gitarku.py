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

#============================================
# Konfigurasi
#============================================

st.set_page_config(
    layout="wide",
    page_title="App Gitarku",
    initial_sidebar_state="collapsed"
)

# konfigurasi server untuk webrtc
KONFIGURASI_RTC = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# daftar pertanyaan kuis
DATA_KUIS = {
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

#=============================================
# Fungsi utility (css, gambar, model)
#=============================================

def load_css(path_file):
    with open(path_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_img_as_base64(path_file):
    """mengubah file gambar jadi string base64 untuk injeksi HTML"""
    with open(path_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_overlay(path_gambar):
    """background menjadi gelap transparan"""
    with open (path_gambar, "rb") as f:
        data = f.read()
    kode_enkripsi = base64.b64encode(data).decode()

    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("data:image/png;base64,{kode_enkripsi}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def load_diagram_chord(nama_chord):
    """Mencari path gambar diagram chord berdasarkan nama."""
    nama_kosong = nama_chord.replace("-Chord", "") 
    kemungkinan_nama = [nama_chord, nama_kosong]
    for nama in kemungkinan_nama:
        for ekstensi in (".png", ".jpg", ".jpeg"):
            path = f"chord_diagrams/{nama}{ekstensi}"
            return path
    return None

@st.cache_resource
def load_yolo_model():
    try:
        path_model = hf_hub_download(
            repo_id="yosidanasmoro03/bestModels",
            filename="best.pt"
        )
        return YOLO(path_model)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# memuat model di awal
model = load_yolo_model()

#===================================================
# Logic WebRTC (untuk memproses video dalam kamera)
#===================================================

class PemrosesKuis(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.target_chord = None
        self.kunci_thread = threading.Lock()
        self.terdeteksi_benar = False

    def update_target(self, target_baru):
        with self.kunci_thread:
            self.target_chord = target_baru
            self.terdeteksi_benar = False
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        gambar = frame.to_ndarray(format="bgr24")

        #Deteksi YOLO
        hasil_deteksi = self.model(gambar, verbose=False, conf=0.5)
        frame_beranotasi = hasil_deteksi[0].plot()

        target_saat_ini = None
        with self.kunci_thread:
            target_saat_ini = self.target_chord

        if len(hasil_deteksi[0].boxes.cls) > 0:
            indeks_terdeteksi = int(hasil_deteksi[0].boxes.cls[0])
            nama_terdeteksi = hasil_deteksi[0].names[indeks_terdeteksi]

            #logic cek jawaban
            if target_saat_ini and nama_terdeteksi == target_saat_ini:
                cv2.rectangle(frame_beranotasi, (50, 50), (450, 150), (0, 255, 0), -1)
                cv2.putText(frame_beranotasi, f"BENAR: {nama_terdeteksi}", (60, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                with self.kunci_thread:
                    self.terdeteksi_benar = True

        return av.VideoFrame.from_ndarray(frame_beranotasi, format="bgr24")

class PemrosesRealtime(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        gambar = frame.to_ndarray(format="bgr24")
        hasil_deteksi = self.model(gambar, verbose=False, conf=0.5)
        frame_beranotasi = hasil_deteksi[0].plot()
        return av.VideoFrame.from_ndarray(frame_beranotasi, format="bgr24")

#===================================================
# Fungsi tampilan halaman
#===================================================

def pertanyaan_kuis_selanjutnya():
    """mengacak soal kuis selanjutnya"""
    teks_soal, target_soal = random.choice(list(DATA_KUIS.items()))
    st.session_state.target_kuis = target_soal
    st.session_state.teks_kuis = teks_soal
    st.session_state["muat_ulang"] = True

# Halaman utama
#==============
def tampilkan_halaman_utama():
    set_background_overlay(r"backgrounds/guitar-unsplash.jpg")

    st.markdown('<h1 style="text-align:center; margin-bottom: 2rem;">🎸 Aplikasi Deteksi Chord Gitar</h1>', unsafe_allow_html=True)

    kol1,kol2,kol3 = st.columns(3, gap="medium")

    # card 1: kuis
    with kol1:
        img_b64 = get_img_as_base64("kuis.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>🎸 Kuis Deteksi Chord</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Kuis"></div>
            <p>Tunjukkan chord yang diminta di depan kamera.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("MULAI KUIS", key="tombol_home_kuis", use_container_width=True):
            st.session_state.menu_pilihan = "🎸 Kuis"
            st.rerun()

    # card 2: realtime
    with kol2:
        img_b64 = get_img_as_base64("realtime.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>🎥 Deteksi Real-time</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Live"></div>
            <p>Deteksi chord bebas menggunakan kamera langsung.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("BUKA KAMERA", key="tombol_home_realtime", use_container_width=True):
            st.session_state.menu_pilihan = "🎥 Real-time"
            st.rerun()

    # card 3: upload gambar
    with kol3:
        img_b64 = get_img_as_base64("upload.png")
        st.markdown(f"""
        <div class="home-card">
            <h3>📷 Upload Gambar</h3>
            <div class="img-container"><img src="data:image/png;base64,{img_b64}" alt="Upload"></div>
            <p>Upload gambar untuk mendeteksi chord yang dimainkan.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("UPLOAD GAMBAR", key="home_up", use_container_width=True):
            st.session_state.menu_pilihan = "📷 Upload"
            st.rerun()

# Halaman fitur kuis
#=====================
def tampilkan_halaman_kuis():
    set_background_overlay(r"backgrounds/acoustic-guitar-dark-surroundings.jpg")

    st.markdown("<h3 style='text-align: center; margin:0; padding:0; color:white;'>🎸 Kuis Deteksi Chord</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    # inisialisasi soal jika belum ada
    if "target_kuis" not in st.session_state:
        pertanyaan_kuis_selanjutnya()
        st.rerun()

    #layout: [spacer, kamera, info, spacer]
    pad_kiri, kol_kamera, kol_info, pad_kanan = st.columns([0.5, 3, 2, 0.5], gap="large")

    with kol_kamera:
        st.write("###### Kamera")
        konteks_webrtc = webrtc_streamer(
            key="quiz_compact",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=KONFIGURASI_RTC,
            video_processor_factory=PemrosesKuis,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with kol_info:
        st.info(f"Target: **{st.session_state.target_kuis}**")
        st.markdown(f"#### {st.session_state.teks_kuis}")

        path_diagram = load_diagram_chord(st.session_state.target_kuis)
        if path_diagram:
            st.image(path_diagram, caption=None, use_container_width=True)

    # update target ke pemroses backend 
    if konteks_webrtc.video_processor:
        konteks_webrtc.video_processor.update_target(st.session_state.target_kuis)

    # logic loop otomatis lanjut soal
    if konteks_webrtc.state.playing:
        placeholder = st.empty()
        while konteks_webrtc.state.playing:
            if konteks_webrtc.video_processor and konteks_webrtc.video_processor.terdeteksi_benar:
                placeholder.success(f"✅ BENAR! {st.session_state.target_kuis}")
                time.sleep(1.0)
                pertanyaan_kuis_selanjutnya()
                st.rerun()
                break
            time.sleep(0.2)

# Halaman fitur deteksi realtime
#=================================
def tampilkan_halaman_realtime():
    set_background_overlay(r"backgrounds/leandro-unsplash.jpg")
    st.markdown("<h3 style='text-align: center; margin:0; color:white;'>🎥 Deteksi Real-time</h3>", unsafe_allow_html=True)

    # layout tengah
    pad_kiri, kol_utama, pad_kanan = st.columns([1,2,1])

    with kol_utama:
        st.write("###### Kamera")
        webrtc_streamer(
            key="realtime_compact",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=KONFIGURASI_RTC,
            video_processor_factory=PemrosesRealtime,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

#halaman upload gambar
def tampilkan_halaman_upload():
    set_background_overlay(r"backgrounds/adi-unsplash.jpg")
    st.markdown("<h3 style= 'text-align: center; margin:0; color:white'>📷 Upload Gambar</h3>", unsafe_allow_html=True)

    kol1, kol2 = st.columns([1,1])

    with kol1:
        uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png"])
    
    with kol2:
        if uploaded_file is not None:
            data_byte = np.frombuffer(uploaded_file.read(), np.uint8)
            gambar = cv2.imdecode(data_byte, cv2.IMREAD_COLOR)

            #prediksi
            if model:
                hasil = model.predict(gambar, verbose=False, conf=0.5)
                frame_beranotasi = hasil[0].plot()
                frame_beranotasi_rgb = cv2.cvtColor(frame_beranotasi, cv2.COLOR_BGR2RGB)

                st.image(frame_beranotasi_rgb, use_container_width=True)

                if len(hasil[0].boxes.cls) > 0:
                    terdeteksi = list(set([hasil[0].names[int(c)] for c in hasil[0].boxes.cls]))
                    st.success(f"Terdeteksi: **{', '.join(terdeteksi)}**")
                else:
                    st.warning("Tidak ada chord terdeteksi.")

#===================================================
# MAIN EXECUTION
#===================================================

def main():
    # load css eksternal
    load_css("styles/styles.css")

    # inisialisasi state menu plihan
    if "menu_pilihan" not in st.session_state:
        st.session_state.menu_pilihan = "🏠 Home"

    # NAVBAR
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    item_navigasi = ["🏠 Home", "🎸 Kuis", "🎥 Real-time", "📷 Upload"]
    kolom_nav = st.columns(len(item_navigasi))

    for i, item in enumerate(item_navigasi):
        with kolom_nav[i]:
            if st.button(item, key=f"nav_main_{i}", use_container_width=True):
                st.session_state.menu_pilihan = item
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # perpindahan halaman
    pilihan_menu_saat_ini = st.session_state.menu_pilihan

    if pilihan_menu_saat_ini == "🏠 Home":
        tampilkan_halaman_utama()
    elif pilihan_menu_saat_ini == "🎸 Kuis":
        tampilkan_halaman_kuis()
    elif pilihan_menu_saat_ini == "🎥 Real-time":
        tampilkan_halaman_realtime()
    elif pilihan_menu_saat_ini == "📷 Upload":
        tampilkan_halaman_upload()

if __name__ == "__main__":
    main()


