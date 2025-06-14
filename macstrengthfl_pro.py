
# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import os
import tempfile
import numpy as np
import pandas as pd
import mediapipe as mp
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

# Setup folders
ASSETS, DATA, PROS = "assets", "data", "pro_clips"
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(PROS, exist_ok=True)
COMMENTS_FILE = os.path.join(DATA, "comments.csv")

# Page config
st.set_page_config(page_title="MacStrengthFL Pro", layout="wide")
bg_img = os.path.join(ASSETS, "background.png")
logo_img = os.path.join(ASSETS, "logo.png")
if os.path.exists(bg_img):
    st.markdown(f"""<style>
    [data-testid="stAppViewContainer"] {{
        background: url('{bg_img}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    </style>""", unsafe_allow_html=True)
if os.path.exists(logo_img):
    logo = Image.open(logo_img)
    st.image(logo, width=200)

# Login
st.sidebar.title("Coach Login")
user = st.sidebar.text_input("Username")
pwd = st.sidebar.text_input("Password", type="password")
if user != "coach" or pwd != "baseball123":
    st.warning("Invalid login.")
    st.stop()

st.title("⚾ MacStrengthFL Pro – AI Swing Analyzer")

# Upload videos
athlete_vid = st.file_uploader("Upload Athlete Swing Video", type=["mp4"], key="athlete")
pro_vid = st.file_uploader("Upload Pro Swing Video (for comparison)", type=["mp4"], key="pro")

# Process videos
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False)
    landmarks_all = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(image)
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            landmarks_all.append(landmarks)
    cap.release()
    return landmarks_all

def draw_pose(frame, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    annotated = frame.copy()
    results = mp_pose.Pose().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return annotated

def score_pose_similarity(landmarks1, landmarks2):
    scores = []
    for l1, l2 in zip(landmarks1, landmarks2):
        if len(l1) == len(l2):
            dist = np.mean([np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(l1, l2)])
            scores.append(dist)
    return round(100 - np.mean(scores)*100, 2) if scores else 0

# Show pose overlays and score
if athlete_vid and pro_vid:
    with tempfile.NamedTemporaryFile(delete=False) as tmp1, tempfile.NamedTemporaryFile(delete=False) as tmp2:
        tmp1.write(athlete_vid.read())
        tmp2.write(pro_vid.read())
        st.video(tmp1.name)
        st.video(tmp2.name)
        st.info("Processing pose data. Please wait...")
        landmarks_athlete = extract_keypoints(tmp1.name)
        landmarks_pro = extract_keypoints(tmp2.name)
        score = score_pose_similarity(landmarks_athlete, landmarks_pro)
        st.success(f"Pose Similarity Score: {score}%")

# Leaderboard
st.subheader("🏆 Athlete Leaderboard")
if os.path.exists(COMMENTS_FILE):
    df = pd.read_csv(COMMENTS_FILE)
    df_scores = df.groupby("Athlete")["Score"].mean().sort_values(ascending=False).reset_index()
    st.dataframe(df_scores)
else:
    st.info("No scores logged yet.")
