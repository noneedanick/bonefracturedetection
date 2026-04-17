from typing import Any
import singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="centered",
)


class Streamlit_YOLOV7(SingleInference_YOLOV7):
    '''
    Streamlit app for bone fracture detection in X-ray images
    '''
    

    def __init__(self,):
        self.logging_main=logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    

    def new_yolo_model(self,img_size,path_yolov7_weights,path_img_i,device_i='cpu'):
        '''
        
        INPUTS:
        img_size,                 #int#   #this is the yolov7 model size, should be square so 640 for a square 640x640 model etc.
        path_yolov7_weights,      #str#   #this is the path to your yolov7 weights 
        path_img_i,               #str#   #path to a single .jpg image for inference (NOT REQUIRED, can load cv2matrix with self.load_cv2mat())

        OUTPUT:
        predicted_bboxes_PascalVOC   #list#  #list of values for detections containing the following (name,x0,y0,x1,y1,score)

        CREDIT
        Please see https://github.com/WongKinYiu/yolov7.git for Yolov7 resources (i.e. utils/models)
        @article{wang2022yolov7,
            title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
            author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
            journal={arXiv preprint arXiv:2207.02696},
            year={2022}
            }
        
        '''
        super().__init__(img_size,path_yolov7_weights,path_img_i,device_i=device_i)
    def main(self):
        st.markdown(
            """
            <style>
            .stApp {
                background-image: url("https://images.unsplash.com/photo-1579548122080-c35fd6820ecb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80");
                background-size: cover;
                background-attachment: fixed;
            }
            /* Center headings */
            .stMainBlockContainer h1,
            .stMainBlockContainer h2,
            .stMainBlockContainer h3 {
                text-align: center;
            }
            /* Center images */
            .stMainBlockContainer [data-testid="stImage"] {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            /* Center buttons */
            .stMainBlockContainer .stButton {
                display: flex;
                justify-content: center;
            }
            .stMainBlockContainer .stButton > button {
                padding: 0.6rem 2.5rem;
                font-size: 1.1rem;
            }
            /* Center file uploader */
            .stMainBlockContainer [data-testid="stFileUploader"] {
                max-width: 500px;
                margin: 0 auto;
            }
            /* Center regular text */
            .stMainBlockContainer .stMarkdown p {
                text-align: center;
            }
            /* Semi-transparent card behind main content for readability */
            .stMainBlockContainer {
                background: rgba(255,255,255,0.85);
                border-radius: 16px;
                padding: 2rem 2rem 1rem 2rem;
            }
            @media (prefers-color-scheme: dark) {
                .stMainBlockContainer {
                    background: rgba(14,17,23,0.88);
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.title(':red[Bone Fracture Detection in Appendicular X-Ray Images]')
        st.subheader(':red[Upload your image and run the model] :rocket:')
        st.write('💁 :green[Use sidebar by clicking left top > button]')

        # --- Sidebar ---
        with st.sidebar:
            st.write("☢️")
            st.write("**:green[This model was trained to detect bone fractures on appendicular skeleton X-ray images]**.") 
            st.write("❗ :green[The model should be used with caution].")
            st.write("❗ :green[It should not be used for medical decision making without an opinion from an expert radiologist.]")
            st.write("☢️")
            st.divider()
            self.conf_selection=st.select_slider('Confidence Threshold',options=[0.05,0.1,0.15,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
            st.divider()
            st.write("**:red[Tips For Getting More Accurate Results]** :bulb:")
            st.write("🟢 If possible convert your DICOM images into jpg or png file format.")
            st.write("🟢 Upload images in a biggest possible resolution (eg. 2k, 3k or 4k).")
            st.write("🟢 Predicting with cropped image of the suspected part may increase the model accuracy (You can experiment with it !).")
            st.write("🟢 Dont try to predict photos taken by cellphone or other camera device from monitor.")
            st.write("🟢 Try to predict on different angle poses of the same subject.") 
            st.write("🟢 Beware of the fact that false positive predictions can happen BUT those can easily differenciated by any clinician !")

        # --- Default image ---
        self.response=requests.get(self.path_img_i)
        self.img_screen=Image.open(BytesIO(self.response.content))
        st.image(self.img_screen, caption=self.capt, use_container_width=True, channels="RGB", output_format="auto")

        # --- Upload + Predict ---
        self.im0=np.array(self.img_screen.convert('RGB'))
        self.load_image_st()

        st.write("")  # spacer
        predictions = st.button(':green[Predict on the image!]')
        if predictions:
            self.predict()
            predictions=False

    def load_image_st(self):
        
        uploaded_img=st.file_uploader(label='⏏️ **:red[Upload an image or try on test image]**')
        if uploaded_img is not None:
            self.img_data=uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0=Image.open(BytesIO(self.img_data))#.convert('RGB')
            self.im0=np.array(self.im0)

            return self.im0
        elif self.im0 is not None:
            return self.im0
        else:
            return None
    
    def predict(self):
        self.conf_thres=self.conf_selection
        
        with st.spinner('Loading image...'):
            self.load_cv2mat(self.im0)
        
        with st.spinner('Making inference...'):
            self.inference()

        self.img_screen=Image.fromarray(self.image).convert('RGB')
        
        self.capt='DETECTED:'
        if len(self.predicted_bboxes_PascalVOC)>0:
            for item in self.predicted_bboxes_PascalVOC:
                name=str(item[0])
                conf=str(round(100*item[-1],2))
                self.capt=self.capt+ ' name='+name+' confidence='+conf+'%, '
        st.image(self.img_screen, caption=self.capt, use_container_width=True, channels="RGB", output_format="auto")
        self.image=None
    

if __name__=='__main__':
    app=Streamlit_YOLOV7()

    #INPUTS for YOLOV7
    img_size=2560
    path_yolov7_weights="weights/best.pt"
    path_img_i="https://github.com/noneedanick/bonefracturedetection/blob/main/test_images/fracture_elbow.jpg?raw=true"
    #INPUTS for webapp
    app.capt="Created by M.D. Murat Yüce and M.D. Gül Gizem Pamuk, Bağcılar Education and Research Hospital, Turkey "
    app.new_yolo_model(img_size,path_yolov7_weights,path_img_i)
    app.conf_thres=0.05
    
    app.load_model() #Load the yolov7 model
    
    app.main()



