import streamlit as st
import tempfile
import cv2
import os
import numpy as np
import subprocess
import random
from tensorflow.keras.models import load_model
from custom_layers import spatial_attention, weighted_bce

MODEL = "lane_detection_final_6.keras"
DATA_DIR = "Data/"

if 'model' not in st.session_state:
    st.session_state['model'] = load_model(MODEL, custom_objects={'weighted_bce': weighted_bce, 'spatial_attention': spatial_attention})

@st.dialog("Video too long")
def stop():
    st.text("Videos longer than 1 min may take too long to process. Please upload another video")
    st.stop()

def process_frame(input_frame):
    original_img = input_frame.copy()
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = cv2.resize(input_frame, (640, 360)) / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    result_frame = st.session_state['model'].predict(input_frame)
    result_frame = np.squeeze(result_frame, axis=0)  # shape: (320, 180, 1)
    result_frame = np.squeeze(result_frame, axis=-1)  # shape: (320, 180)
    result_frame = cv2.resize(result_frame, (1280, 720))
    # Threshold to binary mask (0 or 255)
    result_frame = (result_frame > 0.5).astype(np.uint8) * 255
    # Erode (make lines thinner) before blurring
    erode_kernel = np.ones((3, 3), np.uint8)
    result_frame = cv2.erode(result_frame, erode_kernel, iterations=2)
    # Morphological open - remove small dots
    open_kernel = np.ones((11, 11), np.uint8)
    result_frame = cv2.morphologyEx(result_frame, cv2.MORPH_OPEN, open_kernel)
    for i in range(20):
        result_frame = cv2.GaussianBlur(result_frame, (11, 11), 0)
        _, result_frame = cv2.threshold(result_frame, 127, 255, cv2.THRESH_BINARY)
    # Remove blobs smaller than 1500 area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result_frame, connectivity=8)
    result_frame = np.zeros_like(result_frame)
    min_area = 1500
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            result_frame[labels == label] = 255
    # Convert filtered mask to BGR for overlay
    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_GRAY2BGR)
    # Overlay mask on original image with equal weight
    result = cv2.addWeighted(original_img, 1, result_frame, 1, 0)
    return result

def format_video(path):
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    command = ['ffmpeg', '-y',  # overwrite output if exists
               '-i', path,  # input video path
               '-vcodec', 'libx264',  # video codec avc1 (H.264)
               temp_output  # output video path
              ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(temp_output, 'rb') as f:
        data = f.read()
    os.remove(temp_output)
    os.remove(path)
    return data

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    if duration > 60:
        stop()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_processed = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    output_processed = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    input_video = cv2.VideoWriter(input_processed, fourcc, fps, (1280, 720))
    output_video = cv2.VideoWriter(output_processed, fourcc, fps, (1280, 720))

    progress_bar = st.progress(0)
    progress_text = st.empty()
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        input_video.write(frame)
        output_video.write(process_frame(frame))
        frames_processed += 1
        progress_pct = int((frames_processed / frame_count) * 100)
        progress_bar.progress(progress_pct)
        progress_text.text(f"Processing video: {progress_pct}% completed")

    cap.release()
    input_video.release()
    output_video.release()

    progress_bar.empty()
    progress_text.empty()

    return input_processed, output_processed

st.markdown("""
<style>
.st-key-random p,.st-key-random div {
    height: 40px;
    width: 50px;
    font-size: 60px;
}
.st-key-random p{
    position: relative;
    top: -18px;
    left: -15px;
}
</style>""", unsafe_allow_html=True)


st.title('Lane detection using Attention based CNN model')
col1, col2 = st.columns([10, 1])
col2.text("")
col2.text("")
# Upload video file uploader
video_file = col1.file_uploader('Upload a video file', type=['mp4', 'mov', 'avi', 'mkv'])

# Randomize button
if col2.button("🔀", key="random", help="Choose a random video"):
    video_file = None
    st.markdown("""
    <style>
    .stFileUploader>div{
        display: none;
    }
    </style>""", unsafe_allow_html=True)
    files = [file for file in os.listdir(DATA_DIR)]
    random_video = random.choice(files)
    random_video_path = os.path.join(DATA_DIR, random_video)
    st.info(f"Randomly selected video: {random_video}")
    input_processed, output_processed = process_video(random_video_path)
    if input_processed and output_processed:
        st.subheader("Original Video")
        st.video(format_video(input_processed))
        st.subheader("Processed Video")
        st.video(format_video(output_processed))

# Process uploaded video file
if video_file is not None:
    extension = os.path.splitext(video_file.name)[1]
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    input_temp.write(video_file.read())
    input_temp.flush()
    input_temp.close()

    input_processed, output_processed = process_video(input_temp.name)

    if input_processed and output_processed:
        st.subheader("Original Video")
        st.video(format_video(input_processed))
        st.subheader("Processed Video")
        st.video(format_video(output_processed))


































