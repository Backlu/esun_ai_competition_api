#!/usr/bin/env python
# coding: utf-8

import os, cv2, sys
CUDA_VISIBLE_DEVICES=0
from configparser import ConfigParser
from pathlib import Path
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input as den_preprocess_input
import numpy as np
import pandas as pd
import argparse

# Read config file
config = ConfigParser()
config.optionxform = lambda option: option  # Preserve capital case
config.read(os.path.join(os.path.dirname(__file__), '../CONFIG.ini'))
config_args = dict(config['MODEL_VY'])

weights = Path(config_args['base_dir']).joinpath(config_args['weights'])
label_txt =Path(config_args['base_dir']).joinpath(config_args['label_txt'])

class CWordRecog:
    def __init__(self):
        self.args = None
        self.net = None
        self.names = None

        # self.parse_arguments()
        self.initialize_network()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', type=str, default='', help='Path to use images')
        self.args = parser.parse_args()

    def initialize_network(self):
        # self.weights = '/home/tpe-aa-01/AA/Yvonne/esun/model/model-resnet50-final_tylor.h5'
        self.weights = weights.__str__()
        self.net = load_model(self.weights)
        # label_doc = pd.read_csv('/home/tpe-aa-01/AA/Yvonne/esun/training data dic.txt', header=None, names=['label'])
        label_doc = pd.read_csv(label_txt, header=None, names=['label'])
        label_doc.reset_index(drop=False, inplace=True)
        label_dict = pd.Series(label_doc.label.values,index=label_doc.index).to_dict()
        self.label_dictInv = {k:v for k,v in label_dict.items()}        


    def predict_file(self, image: np.ndarray):
        img = image.load_img(self.args.image, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = self.net.predict(x)[0]
        top_ind = pred.argsort()[::-1][0]
        word = self.label_dictInv[top_ind]
        score = pred[0]
        return word, score

    def pred_word(self, image):
        # img_preprocess = den_preprocess_input(img_bgr_crop)
        img_resize = cv2.resize(image,(224,224))
        # x = image.img_to_array(img_resize)
        x = img_resize
        x = np.expand_dims(x, axis = 0)
        pred = self.net.predict(x)[0]
        top_ind = pred.argsort()[::-1][0]
        word = self.label_dictInv[top_ind]
        score = pred[0]
        return word, score

if __name__== '__main__':

    ocr = ChineseHandWrite.__new__(ChineseHandWrite)
    ocr.__init__()
    word, score = ocr.predict()
    print(word, score)
   
