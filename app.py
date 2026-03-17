# ==========================================================
# 🦅 BIRD INTELLIGENCE SYSTEM
# Modern UI + Image + Camera + Audio + eBird + Analytics
# ==========================================================

import os
import cv2
import csv
import torch
import librosa
import pydeck as pdk
import requests
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import wikipediaapi

from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from transformers import pipeline
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# ==========================================================
# CONFIG
# ==========================================================

EBIRD_API_KEY = "noa4nsprplj4"

st.set_page_config(
    page_title="Bird Intelligence System",
    page_icon="🦅",
    layout="wide"
)

# ==========================================================
# 🎨 PROFESSIONAL UI STYLE
# ==========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f172a,#020617);
    color:white;
}

section[data-testid="stSidebar"] {
    background:#020617;
}

.card {
    background:#0f172a;
    padding:20px;
    border-radius:12px;
    border:1px solid #1e293b;
    box-shadow:0px 6px 20px rgba(0,0,0,0.4);
}

h1,h2,h3 {
    color:#f1f5f9;
}

.stButton>button {
    background:#2563eb;
    color:white;
    border-radius:8px;
}

.stButton>button:hover {
    background:#1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# GPU DETECTION
# ==========================================================

def get_device():

    try:
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
    except:
        pass

    return "cpu","CPU"


device,gpu_name = get_device()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("# 🦅 Bird Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Image Detection",
        "Live Camera",
        "Audio Recognition",
        "Analytics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Device:",device)
st.sidebar.write("GPU:",gpu_name)

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    detector = YOLO("yolov8n.pt")
    detector.to("cpu")

    classifier = pipeline(
        "image-classification",
        model="chriamue/bird-species-classifier",
        device=-1
    )

    birdnet = Analyzer()

    return detector,classifier,birdnet


detector,classifier,birdnet = load_models()

# ==========================================================
# WIKIPEDIA
# ==========================================================

def get_wiki_info(name):

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="bird-ai"
    )

    page = wiki.page(name)

    if page.exists():
        return page.summary

    return "Information unavailable."


# ==========================================================
# BIRD IMAGE GALLERY
# ==========================================================

def get_bird_images(species):

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{species.replace(' ','_')}"

    try:

        r = requests.get(url)

        data = r.json()

        if "thumbnail" in data:

            return [data["thumbnail"]["source"]]

    except:
        pass

    return []


# ==========================================================
# EBIRD OBSERVATIONS
# ==========================================================

def get_bird_observations(species):

    url = "https://api.ebird.org/v2/data/obs/geo/recent"

    headers = {"X-eBirdApiToken":EBIRD_API_KEY}

    params = {
        "lat":20,
        "lng":0,
        "dist":20000
    }

    try:

        r = requests.get(url,headers=headers,params=params)

        data = r.json()

        points = []

        for bird in data:

            if species.lower() in bird["comName"].lower():

                points.append({
                    "lat":bird["lat"],
                    "lon":bird["lng"]
                })

        return pd.DataFrame(points)

    except:

        return pd.DataFrame()


# ==========================================================
# MAP
# ==========================================================

def show_map(species=None):

    if species:

        data = get_bird_observations(species)

        if data.empty:

            data = pd.DataFrame({
                "lat":[20,48,35],
                "lon":[78,10,120]
            })

    else:

        data = pd.DataFrame({
            "lat":[20,48,35],
            "lon":[78,10,120]
        })

    layer = pdk.Layer(
        "ScatterplotLayer",
        data,
        get_position="[lon,lat]",
        get_radius=50000,
        get_fill_color=[255,50,0,160]
    )

    view = pdk.ViewState(
        latitude=20,
        longitude=0,
        zoom=1
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view
        )
    )


# ==========================================================
# DETECTION LOG
# ==========================================================

def log_detection(species,confidence,method):

    os.makedirs("logs",exist_ok=True)

    file = "logs/detections.csv"

    new_file = not os.path.exists(file)

    with open(file,"a",newline="") as f:

        writer = csv.writer(f)

        if new_file:
            writer.writerow(["time","species","confidence","method"])

        writer.writerow([
            datetime.now(),
            species,
            confidence,
            method
        ])

# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.title("🦅 Bird Intelligence System")

    col1,col2,col3 = st.columns(3)

    if os.path.exists("logs/detections.csv"):

        df = pd.read_csv("logs/detections.csv")

        col1.metric("Detections",len(df))
        col2.metric("Species",df["species"].nunique())
        col3.metric("Avg Confidence",round(df["confidence"].mean(),2))

    else:

        col1.metric("Detections",0)
        col2.metric("Species",0)
        col3.metric("Avg Confidence",0)

    st.markdown("---")

    show_map()


# ==========================================================
# IMAGE DETECTION
# ==========================================================

if page == "Image Detection":

    st.header("📷 Bird Image Detection")

    img_file = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

    if img_file:

        img = Image.open(img_file)

        frame = np.array(img)

        st.image(img)

        results = detector(frame)

        boxes = results[0].boxes

        if boxes is None:

            st.warning("No bird detected")

        else:

            for box in boxes:

                cls = int(box.cls[0])

                if cls != 14:
                    continue

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                crop = frame[y1:y2,x1:x2]

                crop = cv2.resize(crop,(224,224))

                crop = Image.fromarray(crop)

                preds = classifier(crop,top_k=3)

                best = preds[0]

                species = best["label"].replace("_"," ").title()

                conf = round(best["score"],3)

                st.markdown('<div class="card">',unsafe_allow_html=True)

                st.subheader(species)

                st.progress(conf)

                st.write(get_wiki_info(species))

                st.markdown('</div>',unsafe_allow_html=True)

                for img in get_bird_images(species):

                    st.image(img,width=250)

                show_map(species)

                log_detection(species,conf,"image")


# ==========================================================
# CAMERA
# ==========================================================

if page == "Live Camera":

    st.header("📹 Live Camera Detection")

    run = st.checkbox("Start Camera")

    frame_window = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret,frame = camera.read()

        if not ret:
            break

        results = detector(frame)

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                if int(box.cls[0]) != 14:
                    continue

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                crop = frame[y1:y2,x1:x2]

                crop = cv2.resize(crop,(224,224))

                crop = Image.fromarray(crop)

                pred = classifier(crop)[0]

                label = pred["label"].replace("_"," ").title()

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,255,0),2)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        frame_window.image(frame)

    camera.release()


# ==========================================================
# AUDIO
# ==========================================================

if page == "Audio Recognition":

    st.header("🎵 Bird Audio Recognition")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav","mp3","ogg"]
    )

    if audio_file:

        st.audio(audio_file)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(audio_file.read())

            path = tmp.name

        audio,sr = librosa.load(path,sr=48000,mono=True)

        wav_path = path+".wav"

        sf.write(wav_path,audio,48000)

        rec = Recording(
            birdnet,
            wav_path,
            min_conf=0.05,
            overlap=1.5
        )

        rec.analyze()

        if rec.detections:

            best = sorted(
                rec.detections,
                key=lambda x:x["confidence"],
                reverse=True
            )[0]

            species = best["common_name"]
            conf = round(best["confidence"],3)

            st.success(species)

            st.progress(conf)

            st.write(get_wiki_info(species))

            show_map(species)

            log_detection(species,conf,"audio")


# ==========================================================
# ANALYTICS
# ==========================================================

if page == "Analytics":

    st.header("📊 Detection Analytics")

    file = "logs/detections.csv"

    if os.path.exists(file):

        df = pd.read_csv(file)

        st.dataframe(df)

        col1,col2,col3 = st.columns(3)

        with col1:
            st.bar_chart(df["species"].value_counts())

        with col2:
            st.bar_chart(df["method"].value_counts())

        with col3:
            st.line_chart(df["confidence"])

    else:

        st.info("No detections yet.")