import os, cv2
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_VISIBLE_DEVICES=0
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.applications.densenet import preprocess_input as den_preprocess_input
import tensorflow as tf
import src.darknet

# Read config file
config = ConfigParser()
config.optionxform = lambda option: option  # Preserve capital case
config.read(os.path.join(os.path.dirname(__file__), '../CONFIG.ini'))
config_args = dict(config['MODEL_V2'])
base_dir = Path(config_args['base_dir'])
yolo_cfg = base_dir.joinpath(config_args['yolo_cfg']).__str__()
yolo_weights = base_dir.joinpath(config_args['yolo_weights']).__str__()
yolo_meta = base_dir.joinpath(config_args['yolo_meta']).__str__()
clf_model = base_dir.joinpath(config_args['clf_model']).__str__()
label_txt = base_dir.joinpath(config_args['label_txt']).__str__()


class CWordRecog:
    #_defaults = {
    #    'yolo_cfg' : 'model/yolov4.cfg',
    #    'yolo_meta' : 'model/yolov4.data',
    #    'yolo_weights' : 'model/yolov4.weights',
    #    'clf_model' : 'model/clf.h5',
    #    'label_txt' : 'doc/training data dic.txt'
    #}
    _defaults = {
        'yolo_cfg' :yolo_cfg,
        'yolo_weights' :yolo_weights,
        'yolo_meta' :yolo_meta,
        'clf_model' :clf_model,
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
        
        #yolo model
        self.network, self.class_names, _ = darknet.load_network(self.yolo_cfg, self.yolo_meta, self.yolo_weights, batch_size=1)
        self.net_w, self.net_h = darknet.network_width(self.network), darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.net_w, self.net_h, 3)
        
        #clf model
        self.clf_model = tf.keras.models.load_model(self.clf_model)
        
        #label dict
        labeldoc_df = pd.read_csv(self.label_txt, header=None, names=['label'])
        labeldoc_df.reset_index(drop=False, inplace=True)
        self.label_dict = pd.Series(labeldoc_df.label.values,index=labeldoc_df.index).to_dict()        

    def __call__(self, img_bgr):
        bbox_list = self.yolo(img_bgr)
        img_bgr_crop = self.crop_word(img_bgr, bbox_list)
        word, score = self.pred_word(img_bgr_crop)
        ret = {'word':word, 'score':score, 'img_bgr_crop':img_bgr_crop,'bbox_list':bbox_list}
        return ret
    
    #------  Classification ------ 
    def pred_word(self, img_bgr_crop):
        img_preprocess = den_preprocess_input(img_bgr_crop)
        img_resize = cv2.resize(img_preprocess,(64,64))
        img_resize = np.expand_dims(img_resize,0)
        pred_prob = self.clf_model.predict(img_resize)[0]
        pred_label = np.argmax(pred_prob)
        #word = self.label_dict[pred_label]
        word = self.label_dict.get(pred_label, 'isnull')
        score = pred_prob[pred_label]
        return word, score
    
    #------ YOLO -------
    def yolo(self, img_bgr):
        img_shape = img_bgr.shape[:2]
        img = img_bgr[:,:,::-1]
        frame_resized = cv2.resize(img, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=self.CONFIDENCE_THRESHOLD)
        detections = sorted(detections, key = lambda x: float(x[1]), reverse=True)
        bbox_list = [list(self.det2bbox(det, img_shape)[2]) for det in detections]
        return bbox_list
    
    def crop_word(self, img, bbox_list):
        img_h,img_w = img.shape[:2]
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
        else:
            img_crop = img
        return img_crop
    
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
    
    def det2bbox(self, detection, img_shape):
        img_h, img_w=img_shape
        r_w, r_h = (img_w/self.net_w, img_h/self.net_h)    
        x, y, w, h = float(detection[2][0]), float(detection[2][1]), float(detection[2][2]), float(detection[2][3])
        x=x*r_w
        w=w*r_w
        y=y*r_h
        h=h*r_h
        x1 = int(round(x - (w / 2)))
        y1 = int(round(y - (h / 2)))
        x2 = int(round(x + (w / 2)))
        y2 = int(round(y + (h / 2))) 
        cx, cy = (x1+x2)//2, (y1+y2)//2
        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)
        obj_class = detection[0]
        obj_score = np.round(float(detection[1]),2)
        obj_bbox = np.array([x1,y1,x2,y2])
        return obj_class, obj_score, obj_bbox
