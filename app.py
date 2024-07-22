import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm')

import streamlit as st
import torch
from ultralytics import YOLO
import tempfile
import cv2
from huggingface_hub import hf_hub_download
import os
import numpy as np
from collections import defaultdict

def run_yolo_inference(weights, source, conf_thres, iou_thres):
    # Load the YOLOv8 model
    model = YOLO(weights)

    # Run inference
    results = model(source, conf=conf_thres, iou=iou_thres)

    return results

def group_detections(detections, tolerance=30):
    """
    Group detections that are close to each other within a given tolerance.
    """
    grouped = defaultdict(list)
    for coords, label in detections:
        found_group = False
        for key in grouped.keys():
            if np.all(np.abs(np.array(coords) - np.array(key)) <= tolerance):
                grouped[key].append(label)
                found_group = True
                break
        if not found_group:
            grouped[coords].append(label)
    return grouped

def summarize_detections(grouped_detections):
    """
    Summarize detections by calculating the average confidence for each group.
    """
    summary = []
    for coords, labels in grouped_detections.items():
        confidences = [float(label.split()[1]) for label in labels]
        avg_confidence = np.mean(confidences)
        summary.append((coords, f"Hardhat {avg_confidence:.2f}"))
    return summary

def main():
    st.title('YOLOv8 Hard Hat Detection App')

    # File upload
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

        # Check if the file was saved correctly
        if os.path.exists(video_path):
            st.text(f"Temporary video file saved at: {video_path}")

            # Download the model from Hugging Face
            st.text("Downloading model from Hugging Face...")
            model_path = hf_hub_download(repo_id="keremberke/yolov8n-hard-hat-detection", filename="best.pt")
            st.text(f"Model downloaded to {model_path}")

            # Run the detection
            st.text('Running detection...')
            results = run_yolo_inference(
                weights=model_path,  # Path to the model
                source=video_path,   # Path to the uploaded video
                conf_thres=0.25,     # Confidence threshold
                iou_thres=0.45       # IoU threshold for NMS
            )

            # Display the resulting video
            st.text('Detection complete. Displaying video...')
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            names = results[0].names if results else []  # Retrieve class names from the first result

            detection_summary = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Render bounding boxes and labels on the frame
                annotated_frame = frame.copy()
                for result in results:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        try:
                            coords = boxes.xyxy[i].cpu().numpy().astype(int)  # Extract the coordinates
                            x1, y1, x2, y2 = coords[:4]
                            confidence = boxes.conf[i].cpu().numpy()
                            class_id = int(boxes.cls[i].cpu().numpy())
                            label = f"{names[class_id]} {confidence:.2f}"

                            # Filter out low-confidence detections
                            if confidence < 0.30:
                                continue

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Collect detection summary with hashable coordinates
                            detection_summary.append((tuple(coords), label))

                        except Exception as e:
                            st.error(f"Error processing box: {e}")

                stframe.image(annotated_frame, channels="BGR")
            cap.release()

            # Display a completion message
            st.success("Video processing complete!")

            # Group and summarize detections
            grouped_detections = group_detections(detection_summary)
            summarized_detections = summarize_detections(grouped_detections)

            # Display detection summary
            st.header("Detection Summary")
            for coords, label in summarized_detections:
                st.write(f"Coordinates: {coords}, Label: {label}")
        else:
            st.error("Failed to save the temporary video file.")

if __name__ == '__main__':
    main()
