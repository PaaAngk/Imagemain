"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
import PIL
from torchvision import models, transforms
import torch
import streamlit as st
import cv2
import re, os, json, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random
import decimal
import shutil
import opendatasets as od
import keras
import math
import scipy
import skimage
import exif
#Image deskew libraries.
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
try:
    from deskew import determine_skew
except:
    from deskew import determine_skew
from typing import Tuple, Union

import segmentation_models as sm
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from scipy.ndimage import interpolation as inter

import warnings
warnings.filterwarnings("ignore")

import base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image:
            linear-gradient(135deg, rgb(201, 177, 251, 0.7), rgb(255, 248, 193, 0.7)), 
            url("data:image/png;base64,%s");
        background-size: cover;
        color: white;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg('cat.jpg')


# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import storage

# cred = credentials.Certificate('./notional-arc-355706-firebase-adminsdk-wsbrr-30edff4cf4.json')
# firebase_admin.initialize_app(cred)

# bucket = storage.bucket('gs://notional-arc-355706.appspot.com')

# import pyrebase

# config = {
#     "apiKey": "AIzaSyDAHWXBKTb-yS9I4Kx4nk11FhAmM5a04Mc",
#     "authDomain": "notional-arc-355706.firebaseapp.com",
#     "projectId": "notional-arc-355706",
#     "databaseURL": "notional-arc-355706.appspot.com",
#     "storageBucket": "https://notional-arc-355706.appspot.com",
#     "messagingSenderId": "204353569112",
#     "appId": "1:204353569112:web:a5d05ddfd037ac1b87e52d",
#     "measurementId": "G-WGM58XVSZX"
# }

# firebase = pyrebase.initialize_app(config)
# storage = firebase.storage()
# storage.child("./test12.jpg").download("test.jpg")
# storage.child("images")


segmentation_model_file = './final_segmentation_model'
faster_rcnn_path = './model_final.pth'


#Function to resize image.
def prod_resize_input(img):
    '''
    Function takes an image and resizes it.
    '''
    img = cv2.resize(img, (224, 224))
    return img.astype('uint8')

#Create function to crop images.
def crop_for_seg(img, bg, mask):
    '''
    Function extracts an image where it overlaps with its binary mask.
    img - Image to be cropped.
    bg - The background on which to cast the image.
    mask - The binary mask generated from the segmentation model.
    '''
    mask = mask.astype('uint8')
    fg = cv2.bitwise_or(img, img, mask=mask) 
    fg_back_inv = cv2.bitwise_or(bg, bg, mask=cv2.bitwise_not(mask))
    New_image = cv2.bitwise_or(fg, fg_back_inv)
    return New_image

def extract_meter(image_to_be_cropped):
    '''
    Function further extracts image such that the meter reading takes up the majority of the image.
    The function finds the edges of the ROI and extracts the portion of the image that contains the entire ROI.
    '''
    where = np.array(np.where(image_to_be_cropped))
    x1, y1, z1 = np.amin(where, axis=1)
    x2, y2, z2 = np.amax(where, axis=1)
    sub_image = image_to_be_cropped.astype('uint8')[x1:x2, y1:y2]
    return sub_image

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    '''
    This function attempts to rotate meter reading images to make them horizontal.
    Its arguments are as follows:
    
    image - The image to be deskewed (in numpy array format).
    angle - The current angle of the image, found with the determine_skew function of the deskew library.
    background - The pixel values of the boarder, either int (default 0) or a tuple.
    
    The function returns a numpy array.
    '''
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def resize_aspect_fit(img, final_size: int):
    '''
    Function resizes the image to specified size.
    
    path - The path to the directory with images.
    final_size - The size you want the final images to be. Should be in int (will be used for w and h).
    write_to - The file you wish to write the images to. 
    save - Whether to save the files (True) or return them.
    '''   
    im_pil = Image.fromarray(img)
    size = im_pil.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im_pil = im_pil.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im_pil, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    new_im = np.asarray(new_im)
    return np.array(new_im)

def prep_for_ocr(img):
    img = resize_aspect_fit(img, 224)
    output_name = 'test_img_for_ocr.jpg'
    cv2.imwrite(output_name, img)
    return output_name

#Converting  mask to rectangle
def rectangle_mask(mask):
    first_el = [len(mask),len(mask)]
    last_el = [0,0]
    for k in range(0, len(mask)-1):
        line = mask[k]
        for j in range(0, len(line)):
            elem = line[j]
            if elem != 0:
                if k<first_el[0] and j<first_el[1]:
                    first_el = [k-5,j-5]
                if k>last_el[0] or j>last_el[1]:
                    last_el = [k,j]
    print("first_el ", first_el, "last_el ", last_el)
    for k in range(0, len(mask)-1):
        line = mask[k]
        for j in range(0, len(line)):
            elem = line[j]
            if k>=first_el[0] and j>=first_el[1] and k<last_el[0] and j<last_el[1]:
                mask[k][j] = 1
    return mask

#Converting  mask to rectangle
def rectangle_mask_for_rgb(mask):
    first_el = [len(mask),len(mask)]
    last_el = [0,0]
    for k in range(0, len(mask)-1):
        line = mask[k]
        for j in range(0, len(line)):
            elem = line[j]
            if np.sum(elem) != 0:
                if k<first_el[0] or j<first_el[1]:
                    first_el = [k,j]
                if k>last_el[0] or j>last_el[1]:
                    last_el = [k,j]

    for k in range(0, len(mask)-1):
        line = mask[k]
        for j in range(0, len(line)):
            elem = line[j]
            if k>=first_el[0] and j>=first_el[1] and k<last_el[0] and j<last_el[1]:
                if np.sum(elem) == 0:
                    mask[k][j] = [255,255,255]
                    
    return mask

#Segment input image.
def segment_input_img(input_img):
    #Convert image from PIL format to opencv
    open_cv_image = np.array(input_img) 
    img = open_cv_image[:, :, :].copy() 

    #Resize image.
    img_small = prod_resize_input(img)
    
    #Open image and get dimensions.
    #input_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    input_w = int(img.shape[1])
    input_h = int(img.shape[0])
    dim = (input_w, input_h)
    
    #Load model, preprocess input, and obtain prediction.
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    img_small = preprocess_input(img_small)
    img_small = img_small.reshape(-1, 224, 224, 3).astype('uint8')
    #we = tf.train.Checkpoint.restore(segmentation_model_file).expect_partial()
    model = tf.keras.models.load_model(segmentation_model_file, custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss, 'iou_score' : sm.metrics.iou_score})
    mask = model.predict(img_small)
    #Change type to uint8 and fill in holes.
    mask = mask.astype('uint8')
    mask = scipy.ndimage.morphology.binary_fill_holes(mask[0, :, :, 0]).astype('uint8')
    #print("Mask: ")
    #mask
    #mask = rectangle_mask(mask)
    #Resize mask to equal input image size.
    mask = cv2.resize(mask, dsize=dim, interpolation=cv2.INTER_AREA)
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((20,20), np.uint8)
    
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    #mask
    #st.image(mask, caption = 'after .')
    #mask = rectangle_mask(mask)
    #st.image(mask, caption = 'before.')
    #Create background array.
    bg = np.zeros_like(img, 'uint8')

    #Get new cropped image and make RGB.
    New_image = crop_for_seg(img, bg, mask)
    New_image = cv2.cvtColor(New_image, cv2.COLOR_BGR2RGB)
    #st.image(New_image, caption = 'New_image.')
    #Extract meter portion.
    extracted = extract_meter(New_image)

    grayscale = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)

    if angle == None:
        angle = 1
    
    st.image(extracted, caption = 'pre rotated.' )
    #rotated = rotate(extracted, angle, (255, 255, 255))
    rotated = rotated1(extracted, angle, (0, 0, 0))

    #rotated = pre_rotation(extracted)
    st.image(rotated, caption = 'rotated.')

    # rotated = rectangle_mask_for_rgb(rotated)
    # st.image(rotated, caption = 'rectangle_mask.')

    return rotated

def pre_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Compute rotated bounding box
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print("pre_rotation angle ",angle )
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def rotated1(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_height, old_width = image.shape[:2]
    print("old_width", old_width)
    print("old_height", old_height)
    print("old_angle", angle)
    print("correct_skew(image)", correct_skew(image))
    if old_width<old_height:
        print("shape")
        print(rotate(image, 10, (0, 0, 0)).shape[:2])
        angle = pre_rotation(image)
        #cur_width = old_width
        #if rotate(image, 5, (0, 0, 0)).shape[:2][0] >  rotate(image, -5, (0, 0, 0)).shape[:2][0]:
        
    else:
        angle_hist = correct_skew(image)
        if abs(angle_hist) < abs(angle):
            angle = angle_hist
    print("!!!!!!!new angle", angle)
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    height = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_height) / 2
    rot_mat[0, 2] += (height - old_width) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def correct_skew(image, delta=1, limit=30):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    return best_angle

def get_reading(file_image):
    '''
    This is the main function for the pipeline. 
    It takes an input image path as its only argument.
    It then car3
    ries out all the necessary steps to extract a meter reading.
    The output is the reading.
    '''
    #Segment image.
    segmented = segment_input_img(file_image)
    
    #Prep image and save path.
    prepped_path = prep_for_ocr(segmented)
    
    st.image(prepped_path, caption = 'prepped_path.', use_column_width = True)
    
    #Class labels.
    labels = ['number', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    #List for storing meter readings.
    list_of_img_reading = []
    
    #Configure model parameters.
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = './model_final.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    predictor = DefaultPredictor(cfg)

    #Read prepped image and obtain prediction.
    im = cv2.imread(prepped_path)
    outputs = predictor(im)
    
    #Find predicted boxes and labels.
    instances = outputs['instances']
    coordinates = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    pred_classes = outputs['instances'].pred_classes.cpu().tolist()
    
    #Obtain list of all predictions and the leftmost x-coordinate for bounding box.
    pred_list = []
    for pred, coord in zip(pred_classes, coordinates):
        pred_list.append((pred, coord[0]))
    
    #Sort the list based on x-coordinate in order to get proper order or meter reading.
    pred_list = sorted(pred_list, key=lambda x: x[1])
    print(pred_list)    
    #Get final order of identified classes, and map them to class value.
    final_predictions = [x[0] for x in pred_list]
    pred_class_names = list(map(lambda x: labels[x], final_predictions))
    
    #Add decimal point to list of digits depending on number of bounding boxes.
    if len(pred_class_names) == 5:
        pass
    else:
        pred_class_names.insert(5, '.')
        
    #Combine digits and convert them into a float.
    print(pred_class_names)
    combine_for_float = "".join(pred_class_names)
    meter_reading = combine_for_float#float()
    return meter_reading

def editOrentation(img):
    pil_image = PIL.Image.open(img)
    with open(img, "rb") as img_file:
        img_file = exif.Image(img_file)
    try:
        orientation = str(img_file.get("orientation"))
    except:
        orientation = ''
        print("Orientation error")
    if orientation == "Orientation.RIGHT_TOP":
        pil_image = pil_image.rotate(-90)
    elif orientation == "Orientation.BOTTOM_RIGHT":
        pil_image = pil_image.rotate(180)
    elif orientation == "Orientation.LEFT_BOTTOM":
        pil_image = pil_image.rotate(90)
    #print("orientation ", str(orientation) )
    return pil_image


# set title of app
st.title("Water Meter Classification")
st.write("")

# enable users to upload images for the model to make predictions
file_image = st.file_uploader("Upload an image", type = "jpg")

file_image = st.camera_input(label = "Or take a pic of meter")

if file_image is not None:
    # display image that user uploaded
    for i in range(6):
        
        file_image = './test%s.jpg'%i
        #file_image = './test5.jpg' 
        image = editOrentation(file_image)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMAGE", file_image," !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        st.image(image, caption = file_image)
        st.write("Just a second ...")
        segmented = segment_input_img(image)    
        #Prep image and save path.
        prepped_path = prep_for_ocr(segmented)
        #st.write("Prediction "+get_reading(image))
    # file_image = './test5.jpg' 
    # image = editOrentation(file_image)
    # st.image(image, caption = 'Uploaded Image.')
    # st.write("Just a second ...")
    # segmented = segment_input_img(image)    
    # #Prep image and save path.
    # prepped_path = prep_for_ocr(segmented)
    # #st.write("Prediction "+get_reading(image))