import json
import cv2
import os
import shutil
import argparse
import itertools
from natsort import natsorted


ap = argparse.ArgumentParser(description='Training')
ap.add_argument('--filter_thresh', default="0.0", type=float, metavar='THRESH')
ap.add_argument('--video_num', default="", type=str, metavar='VIDEO')
ap.add_argument('--data_path', default="19", type=str, metavar='PATH')
ap.add_argument('--custom_test_dir', type=str, metavar='CELL PATH')
ap.add_argument('--conf_score_thres', default=0.99, type=float, metavar='THRESH')
args = ap.parse_args()
filter_thresh = args.filter_thresh
video_num = args.video_num
print(filter_thresh)

def combine_predictions(predictions_json_path, op_path):
    #print(predictions_json_path)
    all_preds = []
    tracking_info_predictions_path = op_path
    for path, diff in predictions_json_path:
        if os.path.exists(path):
            predictions_json = open(path)
            predictions = json.load(predictions_json)
            for pred in predictions:
                pred["diff"] = diff
                all_preds.append(pred)
    all_preds.sort(key = lambda json: json['image_id'])
    with open(tracking_info_predictions_path, "w") as track_info_file:
        json.dump(all_preds, track_info_file)
        
def filter_boxes_size(tracking_info_predictions_path, op_path):
    tracking_info_predictions_filetered_path = op_path
    # filter boxes by size to remove extremely big boxes
    combined_predictions_json = open(tracking_info_predictions_path)
    combined_predictions = json.load(combined_predictions_json)
    combined_predictions_filtered = []
    for pred in combined_predictions:
        width = pred["bbox"][2]
        # print("w", width)
        # print("h", height)
        height = pred["bbox"][3]
        if (width > 35) or (height > 35):
            # print(width)
            # print(height)
            #print("removed")
            #tracking_data.remove(tracked_box)
            continue
        else:
            combined_predictions_filtered.append(pred)
    with open(tracking_info_predictions_filetered_path, "w") as track_info_file:
        json.dump(combined_predictions_filtered, track_info_file)
    specific_file = tracking_info_predictions_filetered_path.split("filtered.json")[0] + "filter_box_size" +".json"
    shutil.copy(tracking_info_predictions_filetered_path, specific_file)

def filter_conf_score(tracking_info_predictions_path, op_path, conf_thresh):
    tracking_info_predictions_filetered_path = op_path
    # filter boxes by size to remove extremely big boxes
    combined_predictions_json = open(tracking_info_predictions_path)
    combined_predictions = json.load(combined_predictions_json)
    combined_predictions_filtered = []
    for pred in combined_predictions:
        score = pred["score"]
        if pred["diff"] == "Motility-low" and score < float(conf_thresh):
            continue
        if pred["diff"] == "Motility-wiggle" and score < float(conf_thresh):
            continue
        elif pred["diff"] == "Motility-mid" and score < float(conf_thresh):
            continue
        elif pred["diff"] == "Motility-high" and score < float(conf_thresh):
            continue
        else:
            combined_predictions_filtered.append(pred)
    with open(tracking_info_predictions_filetered_path, "w") as track_info_file:
        json.dump(combined_predictions_filtered, track_info_file)
    specific_file = tracking_info_predictions_filetered_path.split("filtered.json")[0] + "filter_conf_score" +".json"
    shutil.copy(tracking_info_predictions_filetered_path, specific_file)

def nms_filter(tracking_info_predictions_path, op_path, iou_thresh_nms):
    tracking_info_predictions_filetered_path = op_path
    # filter boxes using nms to remove near duplicate boxes with lower confidence score
    combined_predictions_json = open(tracking_info_predictions_path)
    combined_predictions = json.load(combined_predictions_json)
    nms_filtered_preds = []
    for image_id, preds in itertools.groupby(combined_predictions, key = lambda k:k["image_id"]):
        #print(image_id)
        img_preds = []
        for pred in preds:
            img_preds.append(pred)
        filtered_preds = nms_helper(img_preds, iou_thresh_nms)
        nms_filtered_preds.extend(filtered_preds)
    with open(tracking_info_predictions_filetered_path, "w") as track_info_file:
        json.dump(nms_filtered_preds, track_info_file)
    specific_file = tracking_info_predictions_filetered_path.split("filtered.json")[0] + "filter_nms" +".json"
    shutil.copy(tracking_info_predictions_filetered_path, specific_file)

    
def nms_helper(combined_predictions, iou_thresh_nms):
    iou_thresh_nms=0.70
    final_preds = []
    #get iou of all boxes wrt all boxes
    iou_mat = get_iou_mat_image(combined_predictions, combined_predictions)
    #get matching box pairings
    matching_boxes_iou = get_matching_boxes_iou(combined_predictions, iou_mat, iou_thresh = iou_thresh_nms)
    #run nms while loop only if overlapping boxes exists
    while(len(matching_boxes_iou)>0):
        #sort the list acc to confidence score
        sorted_bbox_list = sorted(combined_predictions, key=lambda k: k["score"])
        #print(sorted_bbox_list[0])
        #print("largest:",sorted_bbox_list[-1])
        #get iou of all boxes wrt to box with max conf score
        iou_mat = get_iou_mat_image(sorted_bbox_list[:-1], [sorted_bbox_list[-1]])
        preds_temp = []
        for index, iou in enumerate(iou_mat):
            if iou[0]<iou_thresh_nms:
                #print(sorted_bbox_list[index])
                preds_temp.append(sorted_bbox_list[index])
        final_preds.append(sorted_bbox_list[-1]) 
        combined_predictions = preds_temp
        iou_mat = get_iou_mat_image(combined_predictions, combined_predictions)
        matching_boxes_iou = get_matching_boxes_iou(combined_predictions, iou_mat, iou_thresh = iou_thresh_nms)
    return final_preds  

#Calculate IoU of 2 bounding boxes
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

     # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_box = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou_box

def get_iou_mat_image(preds_list1, preds_list2 ):
    iou_mat = []
    for pred1 in preds_list1:
        box1 = pred1["bbox"]
        iou_row = []
        for pred2 in preds_list2:
            box2 = pred2["bbox"]
            box1_xyxy = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
            box2_xyxy = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
            iou_boxes = bb_intersection_over_union(box1_xyxy, box2_xyxy)
            iou_row.append(iou_boxes)
        iou_mat.append(iou_row)
    return iou_mat

def get_matching_boxes_iou(preds_list, iou_mat, iou_thresh ):
    matching_iou_boxes = []
    for i in range(0, len(preds_list)):
        iou_row_max = max(iou_mat[i])
        iou_row_max_pred_id = iou_mat[i].index(iou_row_max)
        # print((iou_row_max))
        # print(iou_row_max_pred_id)
        #print(iou_row_max)
        if iou_row_max>iou_thresh:
            matching_iou_boxes.append([i, iou_row_max_pred_id, iou_row_max])
    # print(matching_iou_boxes)  
    #print("Number of matching IOU Ground truth and Predicted boxes: " , len(matching_iou_boxes))
    return matching_iou_boxes

def gen_tracking_data(video_num, predictions_json_path=None, op_path=None):
    # predictions_json_path = f"../NewData-60FPS-Center/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/coco_instances_results.json"
    # predictions_json_path = f"../NewData-60FPS-Center/video{video_num}_feature_optical_flow_median_back_2pyr_18win/test/coco_instances_results.json"
    predictions_json = open(predictions_json_path)
    predictions = json.load(predictions_json)
    try:
        print(predictions[0])
    except:
        print("No predictions")
    tracking_info_predictions_path = op_path + f"./video{video_num}_predictions.json"
    instance_dict = {}
    tracking_info_dict = {}
    instances_image = []
    count = 0
    prev_image_id = predictions[0]['image_id']
    prev_image_name = ""
    image_id = 0
    #print(predictions[-12])
    for image_id, preds in itertools.groupby(predictions, key = lambda k:k["image_id"]):
        #print(image_id)
        instances_image = []
        image_name = str(image_id) + ".tif"
        for prediction in preds:
            score = prediction['score']
            if  score > 0.0: 
                x,y,w,h = prediction['bbox']
                x2 = x + w
                y2 = y + h
                instance_bbox = [x, y, x2, y2]
                instace_dict = {'bbox': instance_bbox, 'labels': prediction['category_id'], 'scores':prediction['score'], 'diff': prediction['diff']}
                instances_image.append(instace_dict)
        tracking_info_dict[image_name] = instances_image
    with open(tracking_info_predictions_path, "w") as track_info_file:
        json.dump(tracking_info_dict, track_info_file)
        
        
data_path =  args.data_path 
combined_pred_path = "coco_instances_results_combined.json"
combined_pred_filtered_path = "coco_instances_results_combined_filtered.json"

video_path = f"data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/"
paths = []
if args.custom_test_dir:
    data_path =  args.custom_test_dir
    video_path = ""
#print(os.path.join(data_path, video_path, f"coco_instances_results_{diff}.json"))

for diff in ["Motility-mid", "Motility-wiggle", "Motility-high" ]:
    paths.append((os.path.join(data_path, video_path, f"coco_instances_results_{diff}.json"), diff))

op_path0 = os.path.join(data_path, video_path, combined_pred_path)
op_path1 = os.path.join(data_path, video_path, combined_pred_filtered_path)

combine_predictions(paths, op_path0)

filter_boxes_size(op_path0, op_path=op_path1)
filter_conf_score(op_path1, op_path=op_path1, conf_thresh=args.conf_score_thres)
nms_filter(op_path1, op_path=op_path1, iou_thresh_nms=filter_thresh)

gen_tracking_data(video_num, op_path1, os.path.join(data_path, video_path))
    