# -*- coding: utf-8 -*-
import os, cv2
CUDA_VISIBLE_DEVICES=0
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input as den_preprocess_input
from tensorflow.keras.applications.nasnet import preprocess_input as nas_preprocess_input
import tensorflow as tf

# Read config file
config = ConfigParser()
config.optionxform = lambda option: option  # Preserve capital case
config.read(os.path.join(os.path.dirname(__file__), '../CONFIG.ini'))
config_args = dict(config['MODEL_V2'])
base_dir = Path(config_args['base_dir'])
yolo_cfg = base_dir.joinpath(config_args['yolo_cfg']).__str__()
yolo_weights = base_dir.joinpath(config_args['yolo_weights']).__str__()
clf_model_path1 = base_dir.joinpath(config_args['clf_cut']).__str__()
clf_model_path2 = base_dir.joinpath(config_args['clf_raw']).__str__()
label_txt = base_dir.joinpath(config_args['label_txt']).__str__()

# +
class CWordRecog:
    _defaults = {
        'yolo_cfg' :yolo_cfg,
        'yolo_weights' :yolo_weights,
        'clf_model_path1' :clf_model_path1,
        'clf_model_path2' :clf_model_path2,
        'label_txt' :label_txt
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"        
     
    def __init__(self, **kwargs):
        #yolo model in ncnn
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)        
        self.CONFIDENCE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4
        self.class_names = ['word']
        net = cv2.dnn.readNet(self.yolo_weights, self.yolo_cfg)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.yolo_model = cv2.dnn_DetectionModel(net)
        self.yolo_model.setInputParams(size=(96, 96), scale=1/255, swapRB=True)
        
        #clf model
        self.clf_model1 = tf.keras.models.load_model(self.clf_model_path1)
        self.clf_model2 = tf.keras.models.load_model(self.clf_model_path2)
        self.imgAug_val1 = ImageDataGenerator(
            samplewise_center=True, samplewise_std_normalization=True,
            preprocessing_function=nas_preprocess_input
        )        
        self.imgAug_val2 = ImageDataGenerator(
            samplewise_center=True, samplewise_std_normalization=True,
            preprocessing_function=den_preprocess_input
        ) 
        
        #label dict
        labeldoc_df = pd.read_csv(self.label_txt, header=None, names=['label'])
        labeldoc_df.reset_index(drop=False, inplace=True)
        self.label_dict = pd.Series(labeldoc_df.label.values,index=labeldoc_df.index).to_dict()
        print('util_v1 ver:0618')

    def __call__(self, img_bgr):
        bbox_list = self.yolo(img_bgr)
        img_bgr_crop, ret = self.crop_word(img_bgr, bbox_list)
        if ret:
            word, score = self.pred_word(img_bgr_crop)
        else:
            word, score = self.pred_word2(img_bgr)
        #ret = {'word':word, 'score':score,'img_bgr_crop':img_bgr_crop,'bbox_list':bbox_list}
        ret = {'word':word, 'score':score}
        return ret
    
    #------  Classification ------ 
    def pred_word2(self, img_bgr):
        img_resize = cv2.resize(img_bgr,(64,64))
        gen = self.imgAug_val2.flow(np.array([img_resize]), batch_size=1, shuffle=False)
        pred_prob = self.clf_model2.predict(gen)[0]
        pred_label = np.argmax(pred_prob)
        score = pred_prob[pred_label]
        if score<0.5:
            word = 'isnull'
        else:
            word = self.label_dict.get(pred_label, 'isnull')
        return word, score
    
    def pred_word(self, img_bgr_crop):
        img_resize = cv2.resize(img_bgr_crop,(64,64))
        gen = self.imgAug_val1.flow(np.array([img_resize]), batch_size=1, shuffle=False)
        pred_prob = self.clf_model1.predict(gen)[0]
        pred_label = np.argmax(pred_prob)
        score = pred_prob[pred_label]
        if score<0.5:
            word = 'isnull'
        else:
            word = self.label_dict.get(pred_label, 'isnull')
        return word, score
    
    #------ YOLO -------
    def yolo(self, img_bgr):
        img_h,img_w = img_bgr.shape[:2]
        _, _, bboxs = self.yolo_model.detect(img_bgr, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        bbox_list = []
        for box in bboxs:
            x1,y1,w,h = box
            new_bbox=[x1,y1,x1+w,y1+h]
            bbox_list.append(new_bbox)
        return bbox_list
    
    def crop_word(self, img, bbox_list):
        img_h,img_w = img.shape[:2]
        img_crop = img
        ret=False
        if len(bbox_list)>0:
            new_wbboxs = self._conbine_detections(bbox_list)
            for i in range(20):
                if len(new_wbboxs)==1:
                    break
                new_wbboxs = self._conbine_detections(new_wbboxs)
            is_multiWord, area_list, sorted_bbox_list = self._isMultiword(new_wbboxs)
            max_bbox = sorted_bbox_list[0]
            x1,y1,x2,y2 = max_bbox
            x1,y1,x2,y2 = max(0,x1-3), max(0,y1-3), min(img_w, x2+3), min(img_h, y2+3)
            img_crop=img[y1:y2,x1:x2]
            ret=True
        return img_crop, ret
    
    def _compute_iou(self, rec1, rec2):
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect))*1.0

    def _combine_bbox(self, bbox1, bbox2):
        new_bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        return new_bbox

    def _conbine_detections(self, bbox_list):
        newbbox_list=[]
        combine_idx=[]
        for k, bbox1 in enumerate(bbox_list):
            if k in combine_idx:
                continue
            has_similar=False
            for i, bbox2 in enumerate(bbox_list):
                if i in combine_idx:
                    continue            
                if k==i:
                    continue
                iou_score = self._compute_iou(bbox1, bbox2)
                if iou_score>0.15:
                    bbox_combine = self._combine_bbox(bbox1, bbox2)
                    newbbox_list.append(bbox_combine)
                    has_similar=True
                    combine_idx.append(i)
                    combine_idx.append(k)
            if has_similar==False:
                newbbox_list.append(bbox1)
        return newbbox_list

    def _get_area(self, bbox):
        x1,y1,x2,y2=bbox
        area = (x2-x1)*(y2-y1)
        return area

    def _isMultiword(self, bbox_list):
        sorted_bbox_list = sorted(bbox_list, key=self._get_area, reverse=True)
        area_list = [ self._get_area(x) for x in sorted_bbox_list]
        max_area = area_list[0]
        max_bbox = sorted_bbox_list[0]
        ratio_list = area_list/max_area
        is_multiWord = (len(np.where(ratio_list>0.7)[0])>1)
        return is_multiWord, area_list, sorted_bbox_list    
    


# -


