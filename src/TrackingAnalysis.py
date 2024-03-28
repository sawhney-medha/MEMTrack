import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import shutil
import pandas as pd
import argparse
from natsort import natsorted


ap = argparse.ArgumentParser(description='Training')
ap.add_argument('--video_num', default="", type=str, metavar='VIDEO')
ap.add_argument('--data_feature_path', default="19", type=str, metavar='PATH')
ap.add_argument('--data_root_path', default="19", type=str, metavar='PATH')
ap.add_argument('--plot_gt', action='store_true')
ap.add_argument('--plot', action='store_true')
ap.add_argument('--custom_test_dir', type=str, metavar='CELL PATH')
ap.add_argument('--min_track_len', default = 60, type=int)
ap.add_argument('--video_map_path', default="data/videomap.txt", type=str, metavar='PATH')
args = ap.parse_args()
video_num = args.video_num

min_track_length = args.min_track_len

#tracking-results:
if args.custom_test_dir:
    op_path = args.custom_test_dir
    video_num= ""
    #print("video_num", video_num)
else:
    op_path = args.data_feature_path +  f"/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/"
tracking_predictions_path = op_path + f"./video{video_num}_tracking_predictions.json"

video_map_path = args.video_map_path
video_map = open(video_map_path,'r',)
header = [x.strip() for x in (video_map.readline().strip()).split(",")]
video_num_id = header.index("video_num")
strain_id = header.index("strain")
strain_map = {}
for line in video_map.readlines():
    line_details = [x.strip() for x in line.split(",")]
    video_id = line_details[video_num_id]
    strain = line_details[strain_id]
    strain_map[video_id] = strain
    
#print(strain_map)
    
with open(tracking_predictions_path) as data_file:
    tracking_data = json.load(data_file)
print("Total tracking data: ", len(tracking_data))
#tracking_data[0]

def filter_boxes_size(tracking_data):
    # filter boxes by size to remove extremely big boxes
    tracking_data_filtered = []
    for tracked_box in tracking_data:
        width = tracked_box["bbox"][2]
        # print("w", width)
        # print("h", height)
        height = tracked_box["bbox"][3]
        if (width > 35) or (height > 35):
            # print(width)
            # print(height)
            #print("removed")
            #tracking_data.remove(tracked_box)
            continue
        else:
            tracking_data_filtered.append(tracked_box)
    return tracking_data_filtered

#tracking_data_filtered = filter_boxes_size(tracking_data)
tracking_data_filtered = tracking_data
#len(tracking_data_filtered)
def groupby_trackid(tracking_data):
    #generate dict to contains tracked boxes per track id
    track_id_details = {}
    for tracked_box in tracking_data:
        #print(tracked_box["track_label"])
        track_label = tracked_box["track_label"]
        #print("track_label", track_label) 
        if track_label not in track_id_details.keys():
            #print("adding to dict")
            track_id_details[track_label] = []
        track_id_details[track_label].append(tracked_box)
    print("----------------------------------------------------------")
    print("sample track id list in dict: ", track_id_details[track_label][:2])
    print("----------------------------------------------------------")
    print("dict track id keys: ", track_id_details.keys())
    print("----------------------------------------------------------")
    return track_id_details

def analyze_track_preds(track_id_details, filter_length=False, min_con_length=11):   
    #get avg frames per track
    #get num of fragments
    #get fragments length
    if filter_length == True:
        track_id_filtered_length = track_id_details.copy()
    num_frames = []
    num_fragments = []
    fragment_lengths = []
    skipped_frames = []
    for track_id in track_id_details.keys():
        num_frames_track = 0
        num_fragments_track = 0
        prev_image_id = 0
        prev_fragment_length = 0 
        new_image = True
        frags_len_track = []
        skip_frames_track = []
        skipped_frames_track = 0
        #print("new track: ", track_id)
        tracked_boxes = track_id_details[track_id]
        #print(tracked_boxes)
        for tracked_box in tracked_boxes:
            #print("box", tracked_box)
            image_id = tracked_box["image_id"]
            num_frames_track += 1
            #print("num frames: ", num_frames_track)
            if new_image == True:
                new_image = False
                num_fragments_track +=1
                prev_image_id = image_id
                fragment_start = image_id

                #print("num fragments: ", num_fragments_track)
            else:
                if prev_image_id + 1 == image_id:
                    prev_fragment_length = prev_image_id - fragment_start + 1
                    prev_image_id +=1

                else:
                    #print("image_id: ", image_id)
                    #print("prev_image_id: ", prev_image_id)
                    skipped_frames_track = image_id-prev_image_id-1
                    skip_frames_track.append(skipped_frames_track)
                    prev_fragment_length = prev_image_id - fragment_start + 1
                    frags_len_track.append(prev_fragment_length)
                    num_fragments_track +=1
                    fragment_start = image_id
                    prev_image_id = image_id
                    #print("num fragments: ", num_fragments_track)
        prev_fragment_length = prev_image_id - fragment_start + 1
        frags_len_track.append(prev_fragment_length)
        if num_fragments_track == 1:
            skip_frames_track.append(skipped_frames_track)
        num_frames.append(num_frames_track)
        if num_frames_track< min_con_length and filter_length is True:
            track_id_filtered_length.pop(track_id, None)
            #print("removing")
        num_fragments.append(num_fragments_track)
        fragment_lengths.append(frags_len_track)
        skipped_frames.extend(skip_frames_track)  
    if filter_length is True:
        return num_frames, num_fragments, fragment_lengths, skipped_frames, track_id_filtered_length
    return num_frames, num_fragments, fragment_lengths, skipped_frames

track_id_details = groupby_trackid(tracking_data_filtered)
num_frames, num_fragments, fragment_lengths, skipped_frames, track_id_filtered_length = analyze_track_preds(track_id_details,
                                                                                                           filter_length=True,min_con_length=min_track_length)
try:
    print("Number of unique track ids: ", len(num_frames))
    print("----------------------------------------------------------")
    print("Average number of frames per track id: ", round(np.mean(num_frames)))
    print("Maximum number of frames per track id: ", np.max(num_frames))
    print("Minimum number of frames per track id: ", np.min(num_frames))
    print("Median number of frames per track id: ", np.median(num_frames))
    print("----------------------------------------------------------")
    print("Average number of fragments per track id: ", round(np.mean(num_fragments)))
    print("Maximum number of fragments per track id: ", np.max(num_fragments))
    print("Minimum number of fragments per track id: ", np.min(num_fragments))
    print("Median number of fragments per track id: ", np.median(num_fragments))
    print("----------------------------------------------------------")
    print("Average number of skipped frames per fragment: ", round(np.mean(skipped_frames)))
    print("Maximum number of skipped frames per fragment: ", np.max(skipped_frames))
    print("Minimum number of skipped frames per fragment: ", np.min(skipped_frames))
except:
    print("Error in print statements")

num_frames, num_fragments, fragment_lengths, skipped_frames = analyze_track_preds(track_id_filtered_length)
try:
    
    print("Number of unique track ids: ", len(num_frames))
    print("----------------------------------------------------------")
    print("Average number of frames per track id: ", round(np.mean(num_frames)))
    print("Maximum number of frames per track id: ", np.max(num_frames))
    print("Minimum number of frames per track id: ", np.min(num_frames))
    print("Median number of frames per track id: ", np.median(num_frames))
    print("----------------------------------------------------------")
    print("Average number of fragments per track id: ", round(np.mean(num_fragments)))
    print("Maximum number of fragments per track id: ", np.max(num_fragments))
    print("Minimum number of fragments per track id: ", np.min(num_fragments))
    print("Median number of fragments per track id: ", np.median(num_fragments))
    print("----------------------------------------------------------")
    print("Average number of skipped frames per fragment: ", round(np.mean(skipped_frames)))
    print("Maximum number of skipped frames per fragment: ", np.max(skipped_frames))
    print("Minimum number of skipped frames per fragment: ", np.min(skipped_frames))
except:
    print("Error in print statements")

all_preds = []
if args.custom_test_dir:
    final_preds_path = args.custom_test_dir
else:
    final_preds_path = args.data_feature_path + f"/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/"
final_preds_path += "coco_instances_results_final.json"
for track_id in track_id_filtered_length.keys():
    for pred in track_id_filtered_length[track_id]:
        all_preds.append(pred)
with open(final_preds_path, "w") as track_info_file:
        json.dump(all_preds, track_info_file)
        
def groupby_imageid(tracking_data):
    #generate dict to contains tracked boxes per track id
    image_id_details = {}
    for track_label in tracking_data:
        for tracked_box in tracking_data[track_label]: 
            image_id = tracked_box["image_id"]
            if image_id not in image_id_details.keys():
            #print("adding to dict")
                image_id_details[image_id] = []
            image_id_details[image_id].append(tracked_box)
    try:
        
        print("----------------------------------------------------------")
        print("sample image id list in dict: ", image_id_details[image_id][:2])
        print("----------------------------------------------------------")
        print("dict image id keys: ", image_id_details.keys())
        print("----------------------------------------------------------")
    except:
        print("Error in print statements")
    return image_id_details

image_id_filtered = groupby_imageid(track_id_filtered_length)

if args.plot :
    
    #create images with labels plot
    data_dir = args.data_path
    data_sub_dirs = glob.glob(f'{data_dir}/*')  
    #print("data dirs: ", data_sub_dirs)
    video_dirs = glob.glob(f'{data_dir}/*/*')  
    for video in video_dirs:
            #print(video)
            if str(video_num) in video:
                print(video)
                video_path = video
                break

            
    img_path =  video_path + f"/frame1/images/"
    if args.custom_test_dir:
        save_path = args.custom_test_dir +"tracklets-filtered/"
    else:
        save_path = args.data_feature_path + f"/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/tracklets-filtered/" 
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)
    
    if args.plot_gt:
        ground_truth_json_path = op_path + "/test_All.json"
        ground_truth_json_path2 = op_path + "/test_Easy+Hard.json"
        if os.path.exists(ground_truth_json_path):
            gt_json = open(ground_truth_json_path)
            ground_truth = json.load(gt_json)
        else: # os.path.exists(ground_truth_json_path2):
            gt_json = open(ground_truth_json_path)
            ground_truth = json.load(gt_json)
        # else:
        #     ground_truth_json_path = args.data_path + f"/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/test_Hard.json"
        #     gt_json = open(ground_truth_json_path)
        #     ground_truth = json.load(gt_json)

        #need frame id, bacteria track id and bbox
        data_path = video_path + f"/frame1/bacteria/"
        bacteria_dirs = os.listdir(data_path)
        print(bacteria_dirs)

        ground_truth_data_bacteria_track_specific = []
        for bacteria in bacteria_dirs:
            #print(bacteria)
            for coord_txt in os.listdir(os.path.join(data_path,bacteria,"xy_coord")):
                #print(coord_txt)
                if not coord_txt.endswith(".ipynb_checkpoints"):
                    frame_id = int(coord_txt.split(".txt")[0])
                    coord_file = open(os.path.join(data_path,bacteria,"xy_coord", coord_txt),'r')
                    line = coord_file.readlines()
                    #print(line)
                    if len(line)>0:
                        x = float(line[0].split(" ")[0])
                        y = float(line[0].split(" ")[1])
                    # print(line)
                    # print(x)
                    # print(y)
                    #print(frame_id)
                        width = height = 30
                        factor_w = 1
                        factor_h = 1
                        x1 = int(x*factor_w - (width // 2))
                        y1 = int(y*factor_h - (height // 2))
                        w=h=30
                        bbox = [x1,y1,w,h]
                        entry = {'image_id': frame_id, 'track_label':int(bacteria), 'bbox':  bbox}
                        ground_truth_data_bacteria_track_specific.append(entry)
        track_id_ground_truth = groupby_trackid(ground_truth_data_bacteria_track_specific)
        ground_truth = groupby_imageid(track_id_ground_truth)



# ground_truth_json_path = f"../NewData-60FPS-Center/CombinationModel/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/test_All.json"
# gt_json = open(ground_truth_json_path)
# ground_truth = json.load(gt_json)
if args.plot :
    for image_id in image_id_filtered:
        #print(image_id)
        newname = save_path + str(image_id) + '.png'
        det_img = cv2.imread(os.path.join(img_path,str(image_id))+".tif")
        det_img_gt_only = det_img.copy()
        det_img_p_only = det_img.copy()

        height, width, channels = det_img.shape
        #print (height, width, channels)
        # plotting filtered predictions
        for tracked_box in image_id_filtered[image_id]:
            bbox = tracked_box["bbox"]
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_cen = x + int(w/2)
            y_cen = y + int(h/2)
            track_label = tracked_box["track_label"]
            cv2.drawMarker(det_img, (x_cen , y_cen),(0,255,255), markerType=cv2.MARKER_CROSS, 
            markerSize=5, thickness=1, line_type=cv2.LINE_AA)

            # print("Y:",y_cen)
            # print("X:",x_cen)
            if x_cen>660:
                print("here")
            if y_cen>490:
                print("here")
            cv2.putText(det_img, '#'+track_label, (x, y-6), 0,0.6,(0,255,255),thickness=1)
        if args.plot_gt:
            #plotting ground truth  
            try:
                for ground_truth_box in ground_truth[image_id]:
                    bbox = ground_truth_box["bbox"]
                    x = bbox[0]
                    y = bbox[1]
                    w = bbox[2]
                    h = bbox[3]
                    x_cen = x + int(w/2)
                    y_cen = y + int(h/2)
                    if x_cen>660:
                        print(image_id)
                        print("here")
                        print("Y:",y_cen)
                        print("X:",x_cen)
                    if y_cen>490:
                        print(image_id)
                        print("here")
                        print("Y:",y_cen)
                        print("X:",x_cen)
                    track_label = ground_truth_box["track_label"]
                    if y_cen>490:
                        print(track_label)
                    cv2.drawMarker(det_img, (x_cen, y_cen),(255,255,0), markerType=cv2.MARKER_TILTED_CROSS, 
                    markerSize=5, thickness=1, line_type=cv2.LINE_AA)
                    cv2.putText(det_img, '#'+str(track_label), (x, y+4), 0,0.6,(255,255,0),thickness=1)
            except:
                pass
        cv2.imwrite(newname,det_img)
    
  
 #image id wise csv
bacteria_count = 0
tracked_raw_data = pd.DataFrame(columns = ['Nr', 'TID', 'PID', 'x [pixel]', 'y [pixel]'])
for image_id in image_id_filtered.keys():
    #print(image_id)
    pid = image_id
    for tracked_box in image_id_filtered[image_id]:
        tracked_bacteria_data = []
        tid = tracked_box["track_label"]
        #print(tid)
        bbox = tracked_box["bbox"]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        x_cen = x + w/2
        y_cen = y + h/2
        bacteria_count +=1
        tracked_bacteria_data ={"Nr" : bacteria_count,'TID' : tid, 'PID' : pid, 
                                'x [pixel]' : x_cen, 'y [pixel]' : y_cen } 
        tracked_raw_data = tracked_raw_data.append(tracked_bacteria_data, ignore_index=True)

#track id wise csa raw data
bacteria_count = 0
tracked_raw_data = pd.DataFrame(columns = ['Nr', 'TID', 'PID', 'x [pixel]', 'y [pixel]'])
for track_id in natsorted(track_id_filtered_length.keys()):
    tid = track_id
    for tracked_box in track_id_filtered_length[track_id]:
        tracked_bacteria_data = []
        pid = tracked_box["image_id"]
        #print(pid)
        bbox = tracked_box["bbox"]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        x_cen = x + w/2
        y_cen = y + h/2
        bacteria_count +=1
        tracked_bacteria_data ={"Nr" : bacteria_count,'TID' : tid, 'PID' : pid, 
                                'x [pixel]' : x_cen, 'y [pixel]' : y_cen } 
        tracked_raw_data = tracked_raw_data.append(tracked_bacteria_data, ignore_index=True)

if len(video_num)>0:        
    tracked_raw_data.to_csv(op_path+ f'TrackedRawData_{min_track_length}_video{video_num}_{strain_map[video_num]}.csv', index=False)
else:
     tracked_raw_data.to_csv(op_path+ f'TrackedRawData_{min_track_length}_video{video_num}.csv', index=False)

#track id wise csa raw data
bacteria_count = 0
tracked_raw_data = pd.DataFrame(columns = ['Nr', 'TID', 'PID', 'x [pixel]', 'y [pixel]'])
for track_id in natsorted(track_id_details.keys()):
    tid = track_id
    for tracked_box in track_id_details[track_id]:
        tracked_bacteria_data = []
        pid = tracked_box["image_id"]
        #print(pid)
        bbox = tracked_box["bbox"]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        x_cen = x + w/2
        y_cen = y + h/2
        bacteria_count +=1
        tracked_bacteria_data ={"Nr" : bacteria_count,'TID' : tid, 'PID' : pid, 
                                'x [pixel]' : x_cen, 'y [pixel]' : y_cen } 
        tracked_raw_data = tracked_raw_data.append(tracked_bacteria_data, ignore_index=True)
        
print("video_num:", video_num)
if len(video_num)>0:
    tracked_raw_data.to_csv(op_path+ f'TrackedRawData_all_video{video_num}_{strain_map[video_num]}.csv', index=False)
else:
     tracked_raw_data.to_csv(op_path+ f'TrackedRawData_all_video{video_num}.csv', index=False)