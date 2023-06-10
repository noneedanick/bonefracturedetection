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
        st.title(':red[Bone Fracture Detection in Appendicular X-Ray Images]')
        st.subheader(""" :red[Upload your image and run the model :rocket:]""")
        st.write(':man-tipping-hand: Use sidebar by clicking left top > button ')
        with st.sidebar:
            st.write(":radioactive_sign:\n"
                     "**:green[This model was trained to detect bone fractures on appendicular skeleton X-ray images]**.\n"  
                     "green[The model should be used with caution].\n"  
                     "green[It should not be used for medical decision making without an opinion from an expert radiologist.]\n"
                     ":radioactive_sign:")
        st.markdown(
            """
        <style>
        .reportview-container .markdown-text-container {
            font-family: monospace;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: black;
        }
        .Widget>label {
            color: green;
            font-family: monospace;
        }
        [class^="st-b"]  {
            color: green;
            font-family: monospace;
        }
        .st-bb {
            background-color: black;
        }
        .st-at {
            background-color: green;
        }
        footer {
            font-family: monospace;
        }
        .reportview-container .main footer, .reportview-container .main footer a {
            color: black;
        }
        header .decoration {
            background-image: None);
        }


        </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <style>
            .stApp {
            background-image: url("https://images.unsplash.com/photo-1579548122080-c35fd6820ecb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80");
            background-size: cover;
            }
            </style>
            """
            ,
            unsafe_allow_html=True
        )
        with st.sidebar:
            text_i_list=[]
            for i,name_i in enumerate(self.names):
                #text_i_list.append(f'id={i} \t \t name={name_i}\n')
                text_i_list.append(f'{i}: {name_i}\n')
            st.selectbox('Classes',tuple(text_i_list))
            self.conf_selection=st.selectbox('Confidence Threshold',tuple([0.05,0.1,0.15,0.25,0.40]))
        
        self.response=requests.get(self.path_img_i)

        self.img_screen=Image.open(BytesIO(self.response.content))

        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        with st.sidebar:
            st.write('**:red[Tips For Getting More Accurate Results :bulb:]**  \n"
                        "* If possible convert your DICOM images into jpg or png file format.  \n"
                        "* Upload images in a biggest possible resolution (eg. 2k, 3k or 4k).  \n"
                        "* Dont try to predict photos taken by cellphone or other camera device from monitor.  \n"
                        "* Try to predict on different angle poses of the same subject.  \n" 
                        "* Beware of the fact that false positive predictions can happen BUT those can easily differenciated by any clinician !  \n")
        self.im0=np.array(self.img_screen.convert('RGB'))
        self.load_image_st()
        
        predictions = st.button(':green[Predict on the image!]')
        if predictions:
            self.predict()
            predictions=False

    def load_image_st(self):
        
        uploaded_img=st.file_uploader(label='**:red[Upload an image or try on test image]**')
        if type(uploaded_img) != type(None):
            self.img_data=uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0=Image.open(BytesIO(self.img_data))#.convert('RGB')
            self.im0=np.array(self.im0)

            return self.im0
        elif type(self.im0) !=type(None):
            return self.im0
        else:
            return None
    
    def predict(self):
        self.conf_thres=self.conf_selection
        
        st.write('Loading image')
        self.load_cv2mat(self.im0)
        
        st.write('Making inference')
        self.inference()

        self.img_screen=Image.fromarray(self.image).convert('RGB')
        
        self.capt='DETECTED:'
        if len(self.predicted_bboxes_PascalVOC)>0:
            for item in self.predicted_bboxes_PascalVOC:
                name=str(item[0])
                conf=str(round(100*item[-1],2))
                self.capt=self.capt+ ' name='+name+' confidence='+conf+'%, '
        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
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



