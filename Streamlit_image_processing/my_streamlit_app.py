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
my_list_of_image_processing = ('Select The Function','Adaptive Thresholding','RGB to Gray','Image Rotation','stitching')

my_list = st.sidebar.selectbox('Select The Function',my_list_of_image_processing)


if my_list == my_list_of_image_processing[1]:
    ImpMyList.adaptive_thresholding()
elif my_list == my_list_of_image_processing[2]:
    ImpMyList.rgb_gray()
elif my_list == my_list_of_image_processing[3]:
    ImpMyList.rotation()
elif my_list == my_list_of_image_processing[4]:
    ImpMyList.stitching()
else:
    pass


