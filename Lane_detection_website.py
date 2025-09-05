import streamlit as st
import tempfile
import cv2
import os
import numpy as np
import subprocess
from tensorflow.keras.models import load_model
from custom_layers import spatial_attention, weighted_bce


MODEL = "lane_detection_final_6.keras"

if 'model' not in st.session_state:
    st.session_state['model'] = load_model(MODEL, custom_objects={'weighted_bce': weighted_bce, 'spatial_attention':spatial_attention})


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
        st.session_state[path] = f.read()

    os.remove(temp_output)
    os.remove(path)
    return st.session_state[path]



# Main code
st.title('Lane detection using Attention based CNN model')
video_file = st.file_uploader('Upload a video file', type=['mp4', 'mov', 'avi', 'mkv'])
if video_file is not None:
    # Save uploaded video temporarily
    extension = os.path.splitext(video_file.name)[1]
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    input_temp.write(video_file.read())
    input_temp.flush()
    input_temp.close()

    # Open input video with OpenCV
    cap = cv2.VideoCapture(input_temp.name)
    if not cap.isOpened():
        st.error("Error opening video file")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0

        if duration > 60:
            stop()

        # 'avc1' codec for H.264 in MP4 container
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Output temp file
        input_processed = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        output_processed = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Create video writer
        input_video = cv2.VideoWriter(input_processed, fourcc, fps, (1280, 720))
        output_video = cv2.VideoWriter(output_processed, fourcc, fps, (1280, 720))

        # Progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        frames_processed = 0

        # Read and write frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720))
            input_video.write(frame)
            output_video.write(process_frame(frame))

            # Update progress bar
            frames_processed += 1
            progress_pct = int((frames_processed / frame_count) * 100)
            progress_bar.progress(progress_pct)
            progress_text.text(f"Processing video: {progress_pct}% completed")

        cap.release()
        input_video.release()
        output_video.release()

        # Remove progress bar
        progress_bar.empty()
        progress_text.empty()

        # Clean up temporary input file
        os.remove(input_temp.name)
        
        st.subheader("Original Video")
        st.video(format_video(input_processed))
        st.subheader("Processed Video")
        st.video(format_video(output_processed))





