import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO("best.pt")

# Function to process each frame and detect hardhats
def detect_hardhats(frame):
    # Run inference
    results = model(frame)
    
    # Annotate the frame with detections
    annotated_frame = frame.copy()
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            try:
                coords = boxes.xyxy[i].cpu().numpy().astype(int)  # Extract the coordinates
                x1, y1, x2, y2 = coords[:4]
                confidence = boxes.conf[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                label = f"{result.names[class_id]} {confidence:.2f}"

                # Filter out low-confidence detections
                if confidence < 0.30:
                    continue

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing box: {e}")

    return annotated_frame

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hardhats in the frame
        annotated_frame = detect_hardhats(frame)

        # Display the resulting frame
        cv2.imshow('Hard Hat Detection', annotated_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
