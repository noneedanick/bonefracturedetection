# YOLOv7 Bone Fracture Detection Deployment with streamlit.

## How to use ?

Freely accessible streamlit app link of this repository : 

https://bonefracturedetection.streamlit.app/

Firstly our model is trained on around 9000 unique X-Ray images (from ethically approved sources) that are annotated by an expert radiologist. Our research showed that the model can increase sensitivity of a clinician for fracture detection on X-Ray images.

Following confusion matrix is the result of our model on test set images:

![model_conf](https://github.com/noneedanick/bonefracturedetection/assets/68031733/8b98c359-8558-4175-a0cd-2cbbb71281fb)

## Tips For Getting More Accurate Results:

- If possible convert your DICOM images into jpg or png file format
- Upload images in a biggest possible resolution (eg. 2k, 3k or 4k)
- Don't try to predict photos taken by cellphone or other camera device from monitor (It is not tested yet :) )
- Try to predict on different angle poses of the same subject
- Beware of the fact that false positive predictions can happen BUT those can easily differenciated by any clinician !


## Thanks to the open source community!
Codes of this repository are adapted from the repository of <a href=https://github.com/stevensmiley1989>stevensmiley1989</a>  https://github.com/stevensmiley1989/STREAMLIT_YOLOV7. 
