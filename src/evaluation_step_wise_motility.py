import random
import os
import cv2
import json
import matplotlib
import argparse
from matplotlib import image, patches
import matplotlib.pyplot as pyplot
import seaborn as sns
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description='Inference')
ap.add_argument('--video_map_path', type=str, metavar='PATH')
ap.add_argument('--data_path',  type=str, metavar='PATH')
ap.add_argument('--video', type=str, metavar='PATH')
ap.add_argument('--all_labels_file',  type=str, metavar='PATH')
args = ap.parse_args()

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

def get_iou_mat_image(detectron_train_file_bbox_list, coco_results_file_bbox_list ):
    # iou_mat = []
    # for gt_box in detectron_train_file_bbox_list:
    #     iou_row = []
    #     for pred_box in coco_results_file_bbox_list:
    #         gt_box_xyxy = [gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
    #         pred_box_xyxy = [pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]]
    #         # print(pred_box_xyxy)
    #         iou_boxes = bb_intersection_over_union(gt_box_xyxy, pred_box_xyxy)
    #         iou_row.append(iou_boxes)
    #     iou_mat.append(iou_row)
    iou_mat = np.zeros((len(detectron_train_file_bbox_list), len(coco_results_file_bbox_list)))
    for i, gt_box in enumerate(detectron_train_file_bbox_list):
        for j, pred_box in enumerate(coco_results_file_bbox_list):
            gt_box_xyxy = [gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
            pred_box_xyxy = [pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]]
            iou_mat[i, j] = bb_intersection_over_union(gt_box_xyxy, pred_box_xyxy)
    return iou_mat

def get_matching_boxes_iou(detectron_train_file_bbox_list, iou_mat, iou_thresh = 0.75):
    matching_iou_boxes = []
    iou_mat = iou_mat.tolist()
    for i in range(0, len(detectron_train_file_bbox_list)):
        if iou_mat[i]:
            iou_row_max = max(iou_mat[i])
            iou_row_max_pred_id = iou_mat[i].index(iou_row_max)
           # print("iou_mat:", iou_mat)           
            # print((iou_row_max))
            # print(iou_row_max_pred_id)
            #print(iou_row_max)
            if iou_row_max>iou_thresh:
                matching_iou_boxes.append([i, iou_row_max_pred_id, iou_row_max])
    # print(matching_iou_boxes)  
    #print("Number of matching IOU Ground truth and Predicted boxes: " , len(matching_iou_boxes))
    return matching_iou_boxes

def compute_mse_matching_boxes(matching_iou_boxes, detectron_train_file_bbox_list, coco_results_file_bbox_list):
    # Compute MSE of fiber intersection based on matching iou-boxes
    mse = []
    iou_sum = 0
    for matching_bbox in matching_iou_boxes:
           # ground truth
            x = detectron_train_file_bbox_list[matching_bbox[0]][0]
            y = detectron_train_file_bbox_list[matching_bbox[0]][1]
            w = detectron_train_file_bbox_list[matching_bbox[0]][2]
            h = detectron_train_file_bbox_list[matching_bbox[0]][3]
            center_x_gt = x + (w/2)
            center_y_gt = y + (h/2)

            # predicted
            x = coco_results_file_bbox_list[matching_bbox[1]][0]
            y = coco_results_file_bbox_list[matching_bbox[1]][1]
            w = coco_results_file_bbox_list[matching_bbox[1]][2]
            h = coco_results_file_bbox_list[matching_bbox[1]][3]
            center_x_pred = x + (w / 2)
            center_y_pred = y + (h / 2)  

            mse.append(((center_x_gt - center_x_pred)**2 + (center_y_gt - center_y_pred)** 2)**0.5)
            iou_sum += matching_bbox[2]
    
   # np.array(matching_iou_boxes)[2].mean()
    if len(matching_iou_boxes)>0:
        return np.array(mse).mean(), iou_sum/len(matching_iou_boxes), mse
    else:
        return np.array(mse).mean(), -1, mse

def nms_predicted_bounding_boxes(pred_bbox_list, pred_bbox_scores, iou_thresh_nms=0.95):
    #NMS of predicted boxes
    #print(len(pred_bbox_list))
    final_tracking_bbox_list =[]
    iou_mat = get_iou_mat_image(pred_bbox_list, pred_bbox_list)
    
    matching_pred_boxes_iou = get_matching_boxes_iou(pred_bbox_list, iou_mat, iou_thresh = iou_thresh_nms)
    while(len(matching_pred_boxes_iou)>0):
        sorted_bbox_list = sorted(zip(pred_bbox_scores, pred_bbox_list))
        iou_mat = get_iou_mat_image(pred_bbox_list, [pred_bbox_list[-1]])
        #print(iou_mat)
        pred_bbox_list_temp = []
        pred_bbox_scores_temp = []
        for index, iou in enumerate(iou_mat):
            if iou[0]<iou_thresh_nms:
                pred_bbox_list_temp.append(pred_bbox_list[index])
                pred_bbox_scores_temp.append(pred_bbox_scores[index])
        # matching_pred_boxes_iou = get_matching_boxes_iou(pred_bbox_list, iou_mat, iou_thresh = iou_thresh_nms)
        # print(matching_pred_boxes_iou)
        final_tracking_bbox_list.append(pred_bbox_list[-1]) #add highest scored bbox to final list
        # matching_pred_boxes_index=[]
        # for bbox in matching_pred_boxes_iou:
        #     matching_pred_boxes_index.append(bbox[0])
        # print("hello",matching_pred_boxes_index)
      
        # for index, bbox in enumerate(pred_bbox_list):
        #     if index not in matching_pred_boxes_index:
        #         pred_bbox_list_temp.append(pred_bbox_list[index])
        #         pred_bbox_scores_temp.append(pred_bbox_scores[index])
        pred_bbox_list = pred_bbox_list_temp
        pred_bbox_scores = pred_bbox_scores_temp
        iou_mat = get_iou_mat_image(pred_bbox_list, pred_bbox_list)
        matching_pred_boxes_iou = get_matching_boxes_iou(pred_bbox_list, iou_mat, iou_thresh = iou_thresh_nms)
        
      
    #print(final_tracking_bbox_list + (pred_bbox_list))
    # print(len(final_tracking_bbox_list))
    # print(len(pred_bbox_list))
    #print(len(final_tracking_bbox_list + (pred_bbox_list)))
    return final_tracking_bbox_list + (pred_bbox_list)

    
def get_statistics(data_path, images_path, coco_eval_path, test_json_path, train_file_name="train.json", detectron_train_file_name = "boardetect_train_coco_format.json",iou_thresh=0.75, bacteria_tp_path="./bacteria-detections/", coco_file = "coco_instances_results.json",print_details=True, store_results_path = None):
    # data_path = "./data_phase_nodes/"
    # images_path = data_path + "images_cropped_resized/"
    # coco_eval_path = "./coco_eval_phase_cropped/"
    num_images = 0
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    mse_image_counter =0
    iou_image_counter =0
    ratio_gt_p = 0
    ratio_p_gt = 0
    mse = 0
    precision = 0
    recall = 0
    iou = 0
    all_img_mse = []
    for filename in sorted(os.listdir(images_path)):  
        # print(filename)
        image_id = filename.split(".tif")[0]
        #print(image_id)
        train_file = json.load(open(data_path + train_file_name,'r'))
        train_file_bbox_list = []
        #print(train_file[int(image_id)]["file_name"])
        if train_file[int(image_id)]["file_name"] == filename:
            for i, annotation in enumerate(train_file[int(image_id)]["annotations"]):
                train_file_bbox_list.append(train_file[int(image_id)]["annotations"][i]["bbox"])
            # print("Number of Ground Truth boxes: ",len(train_file_bbox_list)) 
            # print(train_file_bbox_list[0])

        detectron_train_file = json.load(open(test_json_path + detectron_train_file_name,'r'))
        detectron_train_file_bbox_list = []
        
        
        try:
            for i,annotation in enumerate(detectron_train_file["annotations"]):
                if detectron_train_file["annotations"][i]["image_id"] == int(image_id):
                    detectron_train_file_bbox_list.append(detectron_train_file["annotations"][i]["bbox"])
                #annotation_image_id = detectron_train_file["annotations"][i]["image_id"]
                #print(detectron_train_file["annotations"][i]["bbox"])
        # print(len(detectron_train_file_bbox_list)) 
    #     print(detectron_train_file_bbox_list[0])
            num_ground_truth_boxes = len(detectron_train_file_bbox_list)
        except:
            num_ground_truth_boxes = 0
      

        coco_results_file = json.load(open(coco_eval_path + coco_file,'r'))
        coco_results_file_bbox_list = []
        coco_results_file_bbox_scores = []

        for i,annotation in enumerate(coco_results_file):
            #print(coco_results_file[i])
            #print(image_id)
            if coco_results_file[i]["image_id"] == int(image_id):
                #print(true)
                coco_results_file_bbox_list.append(coco_results_file[i]["bbox"])
                coco_results_file_bbox_scores.append(coco_results_file[i]["score"])
    # print(coco_results_file[i]["bbox"])
    # print(len(coco_results_file_bbox_list))
    # print((coco_results_file_bbox_list))  
                  #print(detectron_train_file["annotations"][i]["bbox"])
        # print("Number of Predicted bounding boxes: ", len(coco_results_file_bbox_list)) 
        # print(coco_results_file_bbox_list[0])

        #coco_results_file_bbox_list = nms_predicted_bounding_boxes(coco_results_file_bbox_list, coco_results_file_bbox_scores ,iou_thresh_nms=0.001)
        
        # Stat 1 - Ratio of boxes Predicted to Groud truth boxes
        num_predicted_boxes = len(coco_results_file_bbox_list) 
        

        num_images = num_images + 1
        if num_ground_truth_boxes>0:
            
            #Ratio of GT to P per image
            ratio_gt_p = ratio_gt_p + (num_ground_truth_boxes/ max(num_predicted_boxes,1))
            ratio_p_gt = ratio_p_gt + (num_predicted_boxes / max(num_ground_truth_boxes,1))
            

            # Stat 2 - MSE of fiber intersections from the matching boxes
            iou_mat = get_iou_mat_image(detectron_train_file_bbox_list, coco_results_file_bbox_list)
            matching_boxes_iou = get_matching_boxes_iou(detectron_train_file_bbox_list, iou_mat, iou_thresh = iou_thresh)
            mse_image, iou_image, mse_list = compute_mse_matching_boxes(matching_boxes_iou, detectron_train_file_bbox_list, coco_results_file_bbox_list)
            if mse_image>=0: #if no predicted boxes or no ground truth boxes then iou is nan
                mse = mse + mse_image
                mse_image_counter += 1
            if iou_image>=0:#if no predicted boxes or no ground truth boxes then mse is nan
                iou += iou_image
                iou_image_counter += 1
        
            true_positive = len(matching_boxes_iou)
            # print("num image: ", num_images)
            # print("Num pred: ", num_predicted_boxes)
            # print("num matched: ", len(matching_boxes_iou))
            os.makedirs(bacteria_tp_path, exist_ok=True)
            coord_file = open(bacteria_tp_path + image_id +".txt",'w')
            coord_file.write(image_id)
            coord_file.write(" ")
            coord_file.write(str(true_positive))
            coord_file.write("\n")
            coord_file.close()
    
            # OLD METHOD OF PRECISION RECALL CALCULATION
            # assumes 0 for no tp
            # precision is average of averages
#             if true_positive>0:
#                 false_positive = num_predicted_boxes - true_positive
#                 false_negative = num_ground_truth_boxes - true_positive

#                 precision += true_positive/(true_positive + false_positive)

#                 recall += true_positive/(true_positive + false_negative)
            all_img_mse.extend(mse_list)

            # UPDATED METHOD
            # precision is total tp / total tp+fp in all images not average of preciosn for all images
        
            false_positive = abs(num_predicted_boxes - true_positive)
            false_negative = abs(num_ground_truth_boxes - true_positive)
            
            tp_sum = tp_sum + true_positive
            fp_sum = fp_sum + false_positive
            fn_sum = fn_sum + false_negative
        
        if num_ground_truth_boxes==0:
            #rint("no gt")
            fp_sum = fp_sum + num_predicted_boxes         
    
    # Stat 1 - Ratio of boxes Predicted to Groud truth boxes
    avg_ratio_gt_p = ratio_gt_p/num_images 
    avg_ratio_p_gt = ratio_p_gt/num_images 
    

    # Stat 2 - MSE of fiber intersections from the matching boxes
    try:
        avg_mse = mse / mse_image_counter
    except:
        avg_mse = 0
    
    
    try:
        avg_iou = iou / iou_image_counter
    except:
        avg_iou = 0
    try:
        avg_prec = tp_sum / (tp_sum + fp_sum)
        avg_recall = tp_sum / (tp_sum + fn_sum)
    except:
        avg_prec = 0
        avg_recall = 0

    if store_results_path:
        result_file = open(store_results_path,'a+')
        result_file.write(str(tp_sum))
        result_file.write(",")
        result_file.write(str(fp_sum))
        result_file.write(",")
        result_file.write(str(fn_sum))
        result_file.write("\n")
        result_file.close()
        
    if print_details:
        # ap = precision/num_images
        # ar = recall/num_images
        print("Average Ground Truth to Predicted Ratio: ", avg_ratio_gt_p)
        print("Average Predicted to Ground Truth Ratio: ", avg_ratio_p_gt)
        print("Average Mean Squared Error of fiber intersections: ", avg_mse)
        print("Average IoU of TP boxes: ", avg_iou)  
        print("TP sum: ", tp_sum)
        print("FP sum: ", fp_sum)
        print("FN sum: ", fn_sum)
        
        print("Average Precision: ", avg_prec)    
        print("Average Recall: ", avg_recall)  

        pyplot.figure(figsize=(10, 3))
        sns.distplot(all_img_mse)
        pyplot.xlabel("Nodal Errors (in px sq)")
        # plt.xlabel("")
        pyplot.grid("on", alpha = 0.3)
        pyplot.show()
    return avg_prec, avg_recall

video_map_path = args.video_map_path #"/alldata/medha/CleanCodeData/Data/videomap.txt"
video_map = open(video_map_path,'r',)
header = [x.strip() for x in (video_map.readline().strip()).split(",")]
video_num_id = header.index("video_num")
strain_id = header.index("strain")
strain_map = {}
for line in video_map.readlines():
    line_details = [x.strip() for x in line.split(",")]
    video_num = line_details[video_num_id]
    strain = line_details[strain_id]
    strain_map[video_num] = strain
strain_map[""] = "all"

#53 54 58 59 60 64 65 69 70 71 75 76 80 81 82 collagen
#85 24 30 35 83 84 agar0.2
#updated gar 19 22 29 30 83 84 
#53 60 64 69 75 82 
video_num = args.video
src_path  = args.data_path #"/alldata/medha/CleanCodeData/DataFeatures/exp_collagen_train_veryhard/"
print(src_path)
video_path = f"data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/"
data_path = os.path.join(src_path,video_path, "test/")
print(data_path)
images_path = data_path + "images/"
coco_eval_path = data_path
test_json_path = data_path
precision = {}
recall = {}
store_file_path_main = src_path + "test_set_results"

# ### Step 1: Detection Individual Models

# In[15]:


try:
    difficulty_level = "Motility-low"
    coco_file=f"coco_instances_results_{difficulty_level}.json"
    detectron_train_file_name = f"boardetect_test_coco_format_{difficulty_level}.json"
    train_file_name=f"test_{difficulty_level}.json"
    store_file_path = store_file_path_main + f"_{difficulty_level}.txt" if store_file_path_main else None
    if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
    #bacteria_tp_path = ./data_video22_feature_optical_flow_median_back_2pyr_18win_background_img/test/1/predicted/"
    # visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
    #                   train_file_name=train_file_name,detectron_train_file_name=detectron_train_file_name, 
    #                   coco_file=coco_file)

    pr, rc = get_statistics(data_path, images_path, coco_eval_path,test_json_path, 
                   train_file_name=train_file_name,detectron_train_file_name = detectron_train_file_name, 
                   iou_thresh=0.1, coco_file=coco_file, print_details=False, store_results_path=store_file_path)
    print("Precision Low: ", pr)
    print("Recall Low: ", rc)
    precision[ "Motility-low"] = pr
    recall[ "Motility-low"] = rc
except:
    precision[ "Motility-low"] = 0.0
    recall[ "Motility-low"] = 0.0
    with open(store_file_path, "r+") as f:
        current_position = previous_position = f.tell()
        while f.readline():
            previous_position = current_position
            current_position = f.tell()
        f.truncate(previous_position)
    f.close()


# In[16]:


try:
    difficulty_level = "Motility-wiggle"
    coco_file=f"coco_instances_results_{difficulty_level}.json"
    detectron_train_file_name = f"boardetect_test_coco_format_{difficulty_level}.json"
    train_file_name=f"test_{difficulty_level}.json"
    store_file_path = store_file_path_main + f"_{difficulty_level}.txt" if store_file_path_main else None
    if store_file_path:
            result_file = open(store_file_path,'a+')
            result_file.write(str(video_num))
            result_file.write(",")
            result_file.close()
    #bacteria_tp_path = ./data_video22_feature_optical_flow_median_back_2pyr_18win_background_img/test/1/predicted/"
    # visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
    #                   train_file_name=train_file_name,detectron_train_file_name=detectron_train_file_name, 
    #                   coco_file=coco_file)

    pr, rc = get_statistics(data_path, images_path, coco_eval_path,test_json_path, 
                   train_file_name=train_file_name,detectron_train_file_name = detectron_train_file_name, 
                   iou_thresh=0.1, coco_file=coco_file, print_details=False, store_results_path=store_file_path)
    print("Precision wiggle: ", pr)
    print("Recall wiggle: ", rc)
    precision["Motility-wiggle"] = pr
    recall["Motility-wiggle"] = rc
except:
    precision["Motility-wiggle"] = 0.0
    recall["Motility-wiggle"] = 0.0
    with open(store_file_path, "r+") as f:
        current_position = previous_position = f.tell()
        while f.readline():
            previous_position = current_position
            current_position = f.tell()
        f.truncate(previous_position)
    f.close()


# In[17]:


try:
    difficulty_level = "Motility-mid"
    coco_file=f"coco_instances_results_{difficulty_level}.json"
    detectron_train_file_name = f"boardetect_test_coco_format_{difficulty_level}.json"
    train_file_name=f"test_{difficulty_level}.json"
    store_file_path = store_file_path_main + f"_{difficulty_level}.txt" if store_file_path_main else None
    if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
    #bacteria_tp_path = ./data_video22_feature_optical_flow_median_back_2pyr_18win_background_img/test/1/predicted/"
    # visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
    #                   train_file_name=train_file_name,detectron_train_file_name=detectron_train_file_name, 
    #                   coco_file=coco_file)

    pr, rc = get_statistics(data_path, images_path, coco_eval_path,test_json_path, 
                   train_file_name=train_file_name,detectron_train_file_name = detectron_train_file_name, 
                   iou_thresh=0.1, coco_file=coco_file, print_details=False, store_results_path=store_file_path)
    print("Precision mid: ", pr)
    print("Recall mid: ", rc)
    precision["Motility-mid"] = pr
    recall["Motility-mid"] = rc
except:
    precision["Motility-mid"] = 0.0
    recall["Motility-mid"] = 0.0
    with open(store_file_path, "r+") as f:
        current_position = previous_position = f.tell()
        while f.readline():
            previous_position = current_position
            current_position = f.tell()
        f.truncate(previous_position)
    f.close()


try:
    difficulty_level = "Motility-high"
    coco_file=f"coco_instances_results_{difficulty_level}.json"
    detectron_train_file_name = f"boardetect_test_coco_format_{difficulty_level}.json"
    train_file_name=f"test_{difficulty_level}.json"
    store_file_path = store_file_path_main + f"_{difficulty_level}.txt" if store_file_path_main else None
    if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
    #bacteria_tp_path = ./data_video22_feature_optical_flow_median_back_2pyr_18win_background_img/test/1/predicted/"
    # visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
    #                   train_file_name=train_file_name,detectron_train_file_name=detectron_train_file_name, 
    #                   coco_file=coco_file)

    pr, rc = get_statistics(data_path, images_path, coco_eval_path,test_json_path, 
                   train_file_name=train_file_name,detectron_train_file_name = detectron_train_file_name, 
                   iou_thresh=0.1, coco_file=coco_file, print_details=False, store_results_path=store_file_path)
    print("Precision high: ", pr)
    print("Recall high: ", rc)
    precision["Motility-high"] = pr
    recall["Motility-high"] = rc
except:
    precision["Motility-high"] = 0.0
    recall["Motility-high"] = 0.0
    with open(store_file_path, "r+") as f:
        current_position = previous_position = f.tell()
        while f.readline():
            previous_position = current_position
            current_position = f.tell()
        f.truncate(previous_position)
    f.close()



    
# ### Step 2: Detection Combination Model

# In[18]:

coco_file = "coco_instances_results_combined.json"


train_file_name=f"test_{args.all_labels_file}.json"
detectron_train_file_name = f"boardetect_test_coco_format_{args.all_labels_file}.json"
store_file_path = store_file_path_main + f"_combined.txt" if store_file_path_main else None
if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
# visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
#                   train_file_name=train_file_name,
#                   detectron_train_file_name = detectron_train_file_name, coco_file=coco_file)
pr, rc = get_statistics(data_path, images_path, coco_eval_path, test_json_path, 
              train_file_name=train_file_name, detectron_train_file_name = detectron_train_file_name,
              iou_thresh=0.1,coco_file=coco_file,print_details=False, store_results_path=store_file_path)
print("Precision Detection combined: ", pr)
print("Recall Detection combined: ", rc)
precision["Combination Model Detection"] = pr
recall["Combination Model Detection"] = rc




# ### Step 3: Filter on Predicted Bacteria Bounding Box Size

# In[19]:


coco_file = "coco_instances_results_combined_filter_box_size.json"
train_file_name="test_All.json"
detectron_train_file_name = "boardetect_test_coco_format_All.json"

train_file_name=f"test_{args.all_labels_file}.json"
detectron_train_file_name = f"boardetect_test_coco_format_{args.all_labels_file}.json"
store_file_path = store_file_path_main + f"_filter_bbox.txt" if store_file_path_main else None
if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
# visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
#                   train_file_name=train_file_name,
#                   detectron_train_file_name = detectron_train_file_name, coco_file=coco_file)
pr, rc = get_statistics(data_path, images_path, coco_eval_path, test_json_path, 
              train_file_name=train_file_name, detectron_train_file_name = detectron_train_file_name,
              iou_thresh=0.1,coco_file=coco_file,print_details=False, store_results_path=store_file_path)
print("Precision Filter Box Size: ", pr)
print("Recall Filter Box Size: ", rc)
precision["Filter Box Size"] = pr
recall["Filter Box Size"] = rc


# ### Step 4: Filter on Predicted Bacteria Confidence Score

# In[20]:


coco_file = "coco_instances_results_combined_filter_conf_score.json"
train_file_name="test_All.json"
detectron_train_file_name = "boardetect_test_coco_format_All.json"

train_file_name=f"test_{args.all_labels_file}.json"
detectron_train_file_name = f"boardetect_test_coco_format_{args.all_labels_file}.json"
store_file_path = store_file_path_main + f"_filter_conf_score.txt" if store_file_path_main else None
if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
# visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
#                   train_file_name=train_file_name,
#                   detectron_train_file_name = detectron_train_file_name, coco_file=coco_file)
pr, rc = get_statistics(data_path, images_path, coco_eval_path, test_json_path, 
              train_file_name=train_file_name, detectron_train_file_name = detectron_train_file_name,
              iou_thresh=0.1,coco_file=coco_file,print_details=False, store_results_path=store_file_path)
print("Precision Filter Conf Score: ", pr)
print("Recall Filter Conf Score: ", rc)
precision["Filter Conf Score"] = pr
recall["Filter Conf Score"] = rc


# ### Step 4: Filter using NMS

# In[21]:


coco_file = "coco_instances_results_combined_filter_nms.json"
train_file_name="test_All.json"
detectron_train_file_name = "boardetect_test_coco_format_All.json"

train_file_name=f"test_{args.all_labels_file}.json"
detectron_train_file_name = f"boardetect_test_coco_format_{args.all_labels_file}.json"

store_file_path = store_file_path_main + f"_filter_nms.txt" if store_file_path_main else None
if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
# visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
#                   train_file_name=train_file_name,
#                   detectron_train_file_name = detectron_train_file_name, coco_file=coco_file)
pr, rc = get_statistics(data_path, images_path, coco_eval_path, test_json_path, 
              train_file_name=train_file_name, detectron_train_file_name = detectron_train_file_name,
              iou_thresh=0.1,coco_file=coco_file,print_details=False, store_results_path=store_file_path)
print("Precision Filter NMS: ", pr)
print("Recall Filter NMS: ", rc)
precision["Filter NMS"] = pr
recall["Filter NMS"] = rc



# ## Step 5: Tracking

# In[22]:


coco_file = f"./video{video_num}_tracking_predictions.json"
train_file_name="test_All.json"
detectron_train_file_name = "boardetect_test_coco_format_All.json"

train_file_name=f"test_{args.all_labels_file}.json"
detectron_train_file_name = f"boardetect_test_coco_format_{args.all_labels_file}.json"
store_file_path = store_file_path_main + f"_tracking.txt" if store_file_path_main else None
if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
# visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
#                   train_file_name=train_file_name,
#                   detectron_train_file_name = detectron_train_file_name, coco_file=coco_file)
pr, rc = get_statistics(data_path, images_path, coco_eval_path, test_json_path, 
              train_file_name=train_file_name, detectron_train_file_name = detectron_train_file_name,
              iou_thresh=0.1,coco_file=coco_file,print_details=False, store_results_path=store_file_path)
print("Precision Tracking: ", pr)
print("Recall Tracking: ", rc)
precision["Tracking"] = pr
recall["Tracking"] = rc




# ### Step 6: Filter on Track length

# In[23]:

coco_file = "coco_instances_results_final.json"
train_file_name="test_All.json"
detectron_train_file_name = "boardetect_test_coco_format_All.json"

train_file_name=f"test_{args.all_labels_file}.json"
detectron_train_file_name = f"boardetect_test_coco_format_{args.all_labels_file}.json"



store_file_path = store_file_path_main + f"_filter_track_length.txt" if store_file_path_main else None
if store_file_path:
        result_file = open(store_file_path,'a+')
        result_file.write(str(video_num))
        result_file.write(",")
        result_file.close()
# visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
#                   train_file_name=train_file_name,
#                   detectron_train_file_name = detectron_train_file_name, coco_file=coco_file)
pr, rc = get_statistics(data_path, images_path, coco_eval_path, test_json_path, 
              train_file_name=train_file_name, detectron_train_file_name = detectron_train_file_name,
              iou_thresh=0.1,coco_file=coco_file,print_details=False, store_results_path=store_file_path)
print("Precision Final: ", pr)
print("Recall Final: ", rc)
precision["Filter Track Length"] = pr
recall["Filter Track Length"] = rc



# #motility analysis

# coco_file = "coco_pred_motile.json"
# train_file_name="test_All.json"
# detectron_train_file_name = "coco_gt_motile.json"
# store_file_path = store_file_path_main + f"_motile.txt" if store_file_path_main else None
# if store_file_path:
#         result_file = open(store_file_path,'a+')
#         result_file.write(str(video_num))
#         result_file.write(",")
#         result_file.close()
# # visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
# #                   train_file_name=train_file_name,
# #                   detectron_train_file_name = detectron_train_file_name, coco_file=coco_file)
# pr, rc = get_statistics(data_path, images_path, coco_eval_path, test_json_path, 
#               train_file_name=train_file_name, detectron_train_file_name = detectron_train_file_name,
#               iou_thresh=0.1,coco_file=coco_file,print_details=False, store_results_path=store_file_path)
# print("Precision: ", pr)
# print("Recall: ", rc)

# coco_file = "coco_pred_non_motile.json"
# train_file_name="test_All.json"
# detectron_train_file_name = "coco_gt_non_motile.json"
# store_file_path = store_file_path_main + f"_non_motile.txt" if store_file_path_main else None
# if store_file_path:
#         result_file = open(store_file_path,'a+')
#         result_file.write(str(video_num))
#         result_file.write(",")
#         result_file.close()
# # visualize_gt_pred(data_path, images_path, coco_eval_path, test_json_path, 
# #                   train_file_name=train_file_name,
# #                   detectron_train_file_name = detectron_train_file_name, coco_file=coco_file)
# pr, rc = get_statistics(data_path, images_path, coco_eval_path, test_json_path, 
#               train_file_name=train_file_name, detectron_train_file_name = detectron_train_file_name,
#               iou_thresh=0.1,coco_file=coco_file,print_details=False, store_results_path=store_file_path)
# print("Precision: ", pr)
# print("Recall: ", rc)

# In[24]:


precision


# In[25]:


recall