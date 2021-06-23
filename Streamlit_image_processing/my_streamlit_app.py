import streamlit as st
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import base64

def get_image_download_link(img):
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href

@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def detectFaceOpenCVDnn(net, frame, framework="caffe", conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False,
    )
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frameHeight / 150)),
                8,
            )
    return frameOpencvDnn, bboxes


#class with all the image processing functions
class ImpMyList():

    def adaptive_thresholding():
        image_file  = st.file_uploader("Choose a file", type =['jpg','jpeg','jfif','png'])
        if image_file is not None:
            place = st.beta_columns(2)
            image = np.array(Image.open(image_file))
            image = cv2.resize(image,(550,550))
            place[0].image(image)
            out_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            out_image = cv2.medianBlur(out_image,3)   
            threshold = st.slider("Value", min_value = 1, max_value = 255, step = 10, value=255) 
            image = cv2.adaptiveThreshold(out_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, threshold, 2)
            place[1].image(image)
            image_d = Image.fromarray(image)
            st.markdown(get_image_download_link(image_d), unsafe_allow_html=True)

    
    def rgb_gray():
        image_file  = st.file_uploader("Choose a rgb file", type =['jpg','jpeg','jfif','png'])
        if image_file is not None:
            place = st.beta_columns(2)
            image = np.array(Image.open(image_file))
            image = cv2.resize(image,(550,550))
            place[0].image(image)
            if st.button("Convert"):
                out_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                place[1].image(out_image)
                image_d = Image.fromarray(out_image)
                st.markdown(get_image_download_link(image_d), unsafe_allow_html=True)

    def face_detection():
        image_file  = st.file_uploader("Choose a rgb file", type =['jpg','jpeg','jfif','png'])
        if image_file is not None:
            net = load_model()
            place = st.beta_columns(2)
            image = np.array(Image.open(image_file))
            image = cv2.resize(image,(350,350))
            place[0].image(image)
            conf_threshold = st.slider("Threshold",min_value = 0.01, max_value = 1.0, step = .01, value=0.5)
            out_image,_ = detectFaceOpenCVDnn(net, image, conf_threshold=conf_threshold)
            place[1].image(out_image)
            image_d = Image.fromarray(out_image)
            st.markdown(get_image_download_link(image_d), unsafe_allow_html=True)

   
    def rotation():
        image_file  = st.file_uploader("Choose a image file", type =['jpg','jpeg','jfif','png'])
        if image_file is not None:
            place = st.beta_columns(2)
            image = Image.open(image_file)
            place[0].image(image)
            degree = st.slider("Degree", min_value = 0, max_value = 360, step = 30, value=90) 
            image = image.rotate(degree)
            place[1].image(image)
            image_d = image
            st.markdown(get_image_download_link(image_d), unsafe_allow_html=True)


    def stitching():
        image_file  = st.file_uploader("Choose first file", type =['jpg','jpeg','jfif','png'])
        if image_file is not None:
            image_file1  = st.file_uploader("Choose second file", type =['jpg','jpeg','jfif','png'])
            place = st.beta_columns(2)
            image1 = np.array(Image.open(image_file))
            image1 = cv2.resize(image1,(550,550))
            place[0].image(image1)
            if image_file1 is not None:
                image2 = np.array(Image.open(image_file1))
                image2 = cv2.resize(image2,(550,550))
                place[1].image(image2)
                if st.button('Concat'):
                    place1 = st.beta_columns(1)
                    image = cv2.hconcat([image1,image2])
                    place1[0].image(image)
                    image_d = Image.fromarray(image)
                    st.markdown(get_image_download_link(image_d), unsafe_allow_html=True)


#Main function starts from here.
st.title("SPEL Web APP USING STREAMLIT")
my_list_of_image_processing = ('Select The Function','Adaptive Thresholding','RGB to Gray','FACE DETECTION','Image Rotation','stitching')

my_list = st.sidebar.selectbox('Select The Function',my_list_of_image_processing)


if my_list == my_list_of_image_processing[1]:
    ImpMyList.adaptive_thresholding()
elif my_list == my_list_of_image_processing[2]:
    ImpMyList.rgb_gray()
elif my_list == my_list_of_image_processing[3]:
    ImpMyList.face_detection()
elif my_list == my_list_of_image_processing[4]:
    ImpMyList.rotation()
elif my_list == my_list_of_image_processing[5]:
    ImpMyList.stitching()
else:
    pass


