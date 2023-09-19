#!/bin/bash
set -e

exp_name="collagen_motility_optical_flow_median_bkg_more_data/"
data_path="/data/medha/Bacteria/DataFeatures/"
data_path+=$exp_name
echo $data_path 

for video_num in  141 148 153 160 167 170  
do
     python ./Scripts/MotilityAnalysis/GenerateTrackingData.py --filter_thresh 0.3 --video_num $video_num --data_path $data_path 
     python ./Scripts/MotilityAnalysis/Tracking.py --video_num $video_num  --data_path $data_path  
     #plot predictions and gt
     python ./Scripts/MotilityAnalysis/TrackingAnalysis.py --video_num $video_num --data_path $data_path #--plot --plot_gt
     # python ./Scripts/MotilityAnalysis/GenerateVideo.py --video_num $video_num --fps 1 --data_path $data_path  
     # python ./Scripts/MotilityAnalysis/GenerateVideo.py --video_num $video_num --fps 60 --data_path $data_path  
done

