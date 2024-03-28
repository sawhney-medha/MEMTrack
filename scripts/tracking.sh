#!/bin/bash
set -e

exp_name="collagen_motility_optical_flow_median_bkg_more_data/"
data_path="/data/medha/Bacteria/DataFeatures/"
data_path+=$exp_name
echo $data_path  
video_map_path="/home/medha/BacteriaDetectionTracking/MEMTrack/data/videomap.txt"
data_root_path="/home/medha/BacteriaDetectionTracking/MEMTrack/data/"

for video_num in  141 148 153 160 167 170  
do
     python src/GenerateTrackingData.py --filter_thresh 0.3 --video_num $video_num --data_path $data_path 
     python src/Tracking.py --video_num $video_num  --data_path $data_path  
     #plot predictions and gt
     python src/TrackingAnalysis.py --video_num $video_num --data_feature_path $data_path --video_map_path $video_map_path --data_root_path $data_root_path #--plot --plot_gt 
     # python src/GenerateVideo.py --video_num $video_num --fps 1 --data_path $data_path  
     # python src/GenerateVideo.py --video_num $video_num --fps 60 --data_path $data_path  
done

