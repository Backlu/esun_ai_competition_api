import os, cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
from tensorflow.keras.applications.densenet import preprocess_input as den_preprocess_input
import tensorflow as tf
import ncnn

class CWordRecog:
    _defaults = {
        'yolo_param' : 'model/esun-opt.param',
        'yolo_bin' : 'model/esun-opt.bin',
        'clf_model' : 'model/clf.h5',
        'label_txt' : 'doc/training data dic.txt'
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
        self.target_size = 224
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.opt.num_threads = 8
        self.net.load_param(self.yolo_param)
        self.net.load_model(self.yolo_bin)
        
        #clf model
        self.clf_model = tf.keras.models.load_model(self.clf_model)
        
        #label dict
        labeldoc_df = pd.read_csv(self.label_txt, header=None, names=['label'])
        labeldoc_df.reset_index(drop=False, inplace=True)
        self.label_dict = pd.Series(labeldoc_df.label.values,index=labeldoc_df.index).to_dict()        

    def __del__(self):
        self.net.clear()
        self.net = None

    def __call__(self, img_bgr):
        bbox_list = self.yolo(img_bgr)
        img_bgr_crop = self.crop_word(img_bgr, bbox_list)
        word, score = self.pred_word(img_bgr_crop)
        ret = {'word':word, 'score':score, 'img_bgr_crop':img_bgr_crop}
        return ret
    
    #------  Classification ------ 
    def pred_word(self, img_bgr_crop):
        img_preprocess = den_preprocess_input(img_bgr_crop)
        img_resize = cv2.resize(img_preprocess,(64,64))
        img_resize = np.expand_dims(img_resize,0)
        pred_prob = self.clf_model.predict(img_resize)[0]
        pred_label = np.argmax(pred_prob)
        word = self.label_dict[pred_label]
        score = pred_prob[pred_label]
        return word, score
    
    #------ YOLO -------
    def yolo(self, img_bgr):
        img_h,img_w = img_bgr.shape[:2]
        mat_in = ncnn.Mat.from_pixels_resize(
            img_bgr, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            img_w, img_h,
            self.target_size, self.target_size,
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        ex = self.net.create_extractor()
        ex.input("data", mat_in)
        ret, mat_out = ex.extract("output")
        bbox_list = []
        for i in range(mat_out.h):
            values = mat_out.row(i)
            label = values[0]
            prob = values[1]
            x1 = values[2] * img_w
            y1 = values[3] * img_h
            x2 = values[4] * img_w
            y2 = values[5] * img_h
            x1 = np.clip(int(x1), 0, img_w)
            y1 = np.clip(int(y1), 0, img_h)
            x2 = np.clip(int(x2), 0, img_w)
            y2 = np.clip(int(y2), 0, img_h)        
            bbox = [x1,y1,x2,y2]
            bbox_list.append(bbox)
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
    

