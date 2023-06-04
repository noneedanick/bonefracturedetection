import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import sys
import torch
import random
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from models.experimental import attempt_load

@st.cache_resource
def load_model():
    device = select_device('0')
    model = attempt_load('weights/best.pt', map_location=device)
    return model

def detect(model, img0, conf_thres=0.05, iou_thres=0.45, classes=None, agnostic_nms=False):
    img = img0.copy()
    img = check_img_size(img, s=model.stride.max())  # check img size
    img = torch.from_numpy(img).to('cuda')
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    return pred

def draw_boxes(img, pred, names):
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=3)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():

    # Add background image from https://c1.wallpaperflare.com/path/664/562/389/xray-doctor-surgeon-hospital-6d9c961c8b0f6f38964e02f9526720e8.jpg
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("https://c1.wallpaperflare.com/path/664/562/389/xray-doctor-surgeon-hospital-6d9c961c8b0f6f38964e02f9526720e8.jpg")
        }
        .sidebar .sidebar-content {
            background: url("https://c1.wallpaperflare.com/path/664/562/389/xray-doctor-surgeon-hospital-6d9c961c8b0f6f38964e02f9526720e8.jpg")
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Bone Fracture Detection in X-Ray Images")
    st.markdown("This app uses a **YOLOv7E6** model trained on a custom dataset to detect fractures in X-ray images.")
    st.markdown("### Upload your image")
    st.markdown("Please upload an image and click the **Detect** button to detect objects in the image.")
    st.markdown("### Or use an example image")
    st.markdown("Click the **Detect** button to detect objects in the image.")
    st.markdown("### Detected objects")
    st.markdown("The detected objects are shown in the image below.")
    
    # Load model
    model = load_model()

    # Load class names
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # Load example images
    example_images = []
    for filename in os.listdir('test_images'):
        example_images.append(os.path.join('images', filename))
    
    # Select image source from sidebar
    image_source = st.sidebar.radio("Select image source:", ("Upload image", "Example image"))
    
    # Detect objects in image and show original and annotated image in main window
   
    if image_source == "Upload image":
        image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            img = load_image(image_file)
            img = np.array(img)
            pred = detect(model, img)
            img = draw_boxes(img, pred, names)
            st.image(img, use_column_width=True)
    else:
        image_file = st.selectbox("Select image:", example_images)
        img = load_image(image_file)
        img = np.array(img)
        pred = detect(model, img)
        img = draw_boxes(img, pred, names)
        st.image(img, use_column_width=True)
    
    # Display class names
    st.markdown("### Class names")
    st.markdown("The following objects can be detected:")
    st.markdown(names)

if __name__ == "__main__":
    main()

