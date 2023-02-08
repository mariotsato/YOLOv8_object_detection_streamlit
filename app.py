
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
#import cv2
import av #strealing video library
import numpy as np
import torch
from time import time
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator


st.title('DEMO: Object detection using YOLOv8 with COCO dataset.')
st.write("The full github code is [here](https://github.com/mariotsato/YOLOv8_object_detection_streamlit)")

########################################################################

#define gpu or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

#declaring the model
model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
model.fuse()

#import the class names    
CLASS_NAMES_DICT = model.model.names

#define the box_annotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)

#define the rtc config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})



def predict(frame):
    results = model(frame)
    return results


def plot_bboxes(results, frame):
    xyxys = []
    confidences = []
    class_ids = []
    # Extract detections for person class
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        if class_id == 0:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
    # Setup detections for visualization
    detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                )
    # Format custom labels
    labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections]
    # Annotate and display frame
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    frame = np.fliplr(frame)
    return frame


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = predict(img)
        img = plot_bboxes(results,img)
        # fps = 1/np.round(end_time - start_time, 2)
        # cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        img = np.fliplr(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

