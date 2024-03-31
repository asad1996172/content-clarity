from dotenv import load_dotenv

load_dotenv()

import os
from urllib.parse import urlparse
import imghdr
import cv2
import moviepy.editor as mp
import numpy as np
import pafy
import streamlit as st
import torch
from tqdm import tqdm
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def add_audio_to_video(video_path, audio_path, output_path, fps=24):
    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(
        output_path, fps=fps, codec="libx264", audio_codec="aac"
    )


def get_best_youtube_video_and_audio_urls(url):
    video_url, audio_url = None, None

    video = pafy.new(url)
    best_video = video.getbestvideo(preftype="mp4")
    best_audio = video.getbestaudio()
    video_url = best_video.url if best_video else None
    audio_url = best_audio.url if best_audio else None

    return video_url, audio_url


def carve_face(image, results, b_mask):
    face_bboxes = results[0].boxes
    height, width, _ = image.shape
    for face_box in face_bboxes:
        x1, y1, x2, y2 = map(int, face_box.xyxy[0])

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        angle = 0
        startAngle = 0
        endAngle = 360

        ellipse_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, center, axes, angle, startAngle, endAngle, 255, -1)

        b_mask[ellipse_mask == 255] = False
    return b_mask


def refine_mask(mask):
    mask_uint8 = mask.astype(np.uint8) * 255

    closing_kernel = np.ones((10, 10), np.uint8)
    closed_mask = cv2.morphologyEx(
        mask_uint8, cv2.MORPH_CLOSE, closing_kernel, iterations=2
    )

    dilation_kernel = np.ones((12, 12), np.uint8)
    dilated_mask = cv2.dilate(closed_mask, dilation_kernel, iterations=2)

    refined_mask = dilated_mask > 127
    return refined_mask


def process_frames(frame, filter_color, face_detection_model):
    img = frame.orig_img
    b_mask = np.zeros(img.shape[:2], np.uint8)
    for _, c in enumerate(frame):
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        b_mask = cv2.drawContours(b_mask, [contour], -1, 1, cv2.FILLED)

    if b_mask.sum() == 0:
        return img

    b_mask = b_mask.astype(bool)
    b_mask = refine_mask(b_mask)

    b_mask = carve_face(img, face_detection_model(img), b_mask)

    img[b_mask] = filter_color
    return img


def get_input_source_info(input_source):
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)

    cap.release()
    return total_frames, fps, frame_size


def process_input(
    processed_image_placeholder,
    is_youtube_video,
    input_source,
    output_path,
    filter_color,
    segmentation_model,
    face_detection_model,
    progress_bar,
):
    if urlparse(input_source).hostname in (
        "www.youtube.com",
        "youtube.com",
        "youtu.be",
    ):
        input_source, audio_source = get_best_youtube_video_and_audio_urls(input_source)
        is_youtube_video = True

    total_frames, fps, frame_size = get_input_source_info(input_source)

    results = segmentation_model.predict(
        input_source,
        stream=True,
        conf=0.25,
        iou=0.45,
        device=DEVICE,
        classes=[0],
        verbose=False,
    )
    output = []
    for i, result in tqdm(
        enumerate(results, start=1), "Processing frame(s): ", total=total_frames
    ):
        progress_bar.progress(int(100 * i / total_frames))
        processed_frame = process_frames(result, filter_color, face_detection_model)
        if i % 100 == 1:
            processed_image_placeholder.image(processed_frame, use_column_width=True)
        output.append(processed_frame)

    os.makedirs(output_path, exist_ok=True)
    if len(output) == 1:
        output_file_path = os.path.join(output_path, "output.jpg")
        cv2.imwrite(output_file_path, output[0])
        print("Image saved to:", os.path.join(output_path, "output.jpg"))
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        processed_video_path = os.path.join(output_path, "processed_video.mp4")
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, frame_size)
        for frame in output:
            out.write(frame)
        out.release()

        if is_youtube_video:
            output_file_path = os.path.join(output_path, "yt_output.mp4")
            add_audio_to_video(
                processed_video_path, audio_source, output_file_path, fps
            )
            os.remove(processed_video_path)
            print("Video saved to:", output_file_path)
        else:
            output_file_path = os.path.join(output_path, "output.mp4")
            add_audio_to_video(
                processed_video_path, input_source, output_file_path, fps
            )
            os.remove(processed_video_path)
            print("Video saved to:", output_file_path)
    return output_file_path


if __name__ == "__main__":
    st.sidebar.title("Options")

    upload_folder = "uploaded_files"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    uploaded_file = st.sidebar.file_uploader(
        "Load file (Image/Video)", type=["png", "jpg", "jpeg", "mp4"]
    )

    segmentation_model_name = st.sidebar.selectbox(
        "Select Segmentation Model",
        [
            "yolov8n-seg.pt",
            "yolov8l-seg.pt",
            "yolov8m-seg.pt",
            "yolov8s-seg.pt",
            "yolov8x-seg.pt",
        ],
    )
    face_detection_model_name = st.sidebar.selectbox(
        "Select Face Detection Model", ["yolov8n-face.pt"]
    )
    progress_bar = st.progress(0)
    progress_bar.empty()

    original_image_label = st.empty()
    original_image_placeholder = st.empty()
    processed_image_label = st.empty()
    processed_image_placeholder = st.empty()
    if st.sidebar.button("Submit", key="submit_button", use_container_width=True):
        segmentation_model = YOLO(f"yolo_models/{segmentation_model_name}").to(DEVICE)
        face_detection_model = YOLO(f"yolo_models/{face_detection_model_name}").to(
            DEVICE
        )
        filter_color = (0, 0, 0)

        original_image_label.empty()
        original_image_placeholder.empty()
        processed_image_label.empty()
        processed_image_placeholder.empty()
        
        if uploaded_file is not None:
            file_path = os.path.join(upload_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            output_file_path = process_input(
                processed_image_placeholder,
                False,
                file_path,
                "outputs/",
                filter_color,
                segmentation_model,
                face_detection_model,
                progress_bar,
            )

            if imghdr.what(output_file_path):
                original_image_label.header("Original Image")
                original_image_placeholder.image(file_path)
                
                processed_image_label.header("Processed Image")
                processed_image_placeholder.image(output_file_path)

                with open(output_file_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Processed Image",
                        data=file,
                        file_name="processed_image.png",
                        mime="image/png"
                    )
                
            else:
                original_image_label.header("Original Video")
                video_file = open(file_path, "rb")
                video_bytes = video_file.read()
                original_image_placeholder.video(video_bytes)
                
                processed_image_label.header("Processed Video")
                video_file = open(output_file_path, "rb")
                video_bytes = video_file.read()
                processed_image_placeholder.video(video_bytes)

                with open(output_file_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video.png",
                        mime="video/mp4"
                    )
        else:
            st.info(
                "Please upload Image/Video or provide a YouTube link and then press Submit !"
            )

        progress_bar.progress(100)
        progress_bar.empty()

    else:
        st.info(
            "Please upload Image/Video or provide a YouTube link and then press Submit !"
        )