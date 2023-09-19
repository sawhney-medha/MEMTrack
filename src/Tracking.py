import json
import matplotlib 
import matplotlib.pylab as plt
from contextlib import suppress
from sort import Sort
import collections
from pprint import pprint
from sort import *
import os
import cv2
import numpy as np
import shutil
import argparse
from natsort import natsorted

ap = argparse.ArgumentParser(description='Training')
ap.add_argument('--video_num', default="", type=str, metavar='VIDEO')
ap.add_argument('--data_path', default="19", type=str, metavar='PATH')
ap.add_argument('--custom_test_dir', type=str, metavar='CELL PATH')
ap.add_argument('--max_age', type=int, metavar='CELL PATH', default=25)
ap.add_argument('--max_interpolation', type=int, metavar='CELL PATH', default=25)
args = ap.parse_args()
video_num = args.video_num

max_age = args.max_age
max_interpolation = args.max_interpolation

def track_bacteria(video_num, max_age, max_interpolation, op_path=None,):
    #mot_tracker = Sort(max_age = 25, min_hits=0, iou_threshold=0.05, max_interpolation=25)
    mot_tracker = Sort(max_age = max_age, min_hits=0, iou_threshold=0.05, max_interpolation=max_interpolation)
    #video_num = 29
    if args.custom_test_dir:
        op_path = args.custom_test_dir
        video_num = ""
        # jsonpath = op_path + f"./video_predictions.json"
        # save_path = op_path + f"./video-tracklets/"
    else:
        op_path = args.data_path +  f"/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/"
    
    jsonpath = op_path + f"./video{video_num}_predictions.json"
    save_path = op_path + f"./video{video_num}-tracklets/"
    tracking_predictions_path = op_path +f"./video{video_num}_tracking_predictions.json"
    img_path = op_path + "/images/"
    
   

    with open(jsonpath) as data_file:
        data = json.load(data_file)
    odata = collections.OrderedDict(sorted(data.items()))
    print(jsonpath)
    
    #img_path = f"../NewData-60FPS-Center/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/images"
    
    print(img_path)
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)

    tracking_predictions = []

    for key in natsorted(odata.keys()): 
        arrlist = []
        det_img = cv2.imread(os.path.join(img_path, key))
        height, width, channels = det_img.shape
        overlay = det_img.copy()
        det_result = data[key] 

        for info in det_result:
            bbox = info['bbox']
            #add filter bbox for size
            labels = info['labels']
            scores = info['scores']
            templist = bbox+[scores]       
            arrlist.append(templist)

        track_bbs_ids = mot_tracker.update(np.array(arrlist))

        mot_imgid = key.replace('.tif','')
        newname = save_path + mot_imgid + '_mot.jpg'
        #print(mot_imgid)

        for j in range(track_bbs_ids.shape[0]):  
            ele = track_bbs_ids[j, :]
            x = int(ele[0])
            y = int(ele[1])
            x2 = int(ele[2])
            y2 = int(ele[3])
            x_cen = x + int((x2-x)/2)
            y_cen = y + int((y2-y)/2)
            track_label = str(int(ele[4]))
            if x_cen>= width:
                continue
            if y_cen>= height:
                continue
            #cv2.rectangle(det_img, (x, y), (x2, y2), (0, 255, 255), 4)
            # cv2.line(det_img, (x, y), (x2, y2), (0, 255, 255), 1, )
            # cv2.line(det_img, (x2, y), (x, y2), (0, 255, 255), 1)
            cv2.drawMarker(det_img, (x_cen + 5, y_cen + 5),(0,255,255), markerType=cv2.MARKER_CROSS, 
            markerSize=10, thickness=1, line_type=cv2.LINE_AA)
            cv2.putText(det_img, '#'+track_label, (x, y-4), 0,0.4,(0,255,255),thickness=1)
            w = x2 - x
            h = y2 - y
            bbox = [x, y, w, h]
            instance_prediction = {'image_id': int(mot_imgid) , 'category_id': 0, 'bbox': bbox, 'score':-1, 'track_label': track_label}
            tracking_predictions.append(instance_prediction)

        cv2.imwrite(newname,det_img)
        
        
    with open(tracking_predictions_path, "w") as track_preds_file:
        json.dump(tracking_predictions, track_preds_file)
    

track_bacteria(video_num, max_age, max_interpolation)