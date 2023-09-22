#!/bin/bash
set -e
exp_name="collagen_motility_inference" # exp name
data_path="/home/medha/BacteriaDetectionTracking/MEMTrack/DataFeatures/" # feature directory path
data_path+=$exp_name
echo $data_path 


#path to saved models
low_motility_model_path="/home/medha/BacteriaDetectionTracking/MEMTrack/models/motility/low/collagen_optical_flow_median_bkg_more_data_90k/" 
wiggle_motility_model_path="/home/medha/BacteriaDetectionTracking/MEMTrack/models/motility/wiggle/collagen_optical_flow_median_bkg_more_data_90k/" 
mid_motility_model_path="/home/medha/BacteriaDetectionTracking/MEMTrack/models/motility/mid/collagen_optical_flow_median_bkg_more_data_90k/" 
high_motility_model_path="/home/medha/BacteriaDetectionTracking/MEMTrack/models/motility/high/collagen_optical_flow_median_bkg_more_data_90k/"  


#update test video numbers from video map
for video_num in  2 
do
   
   #To genearate testing files for all motilities
      python src/inferenceBacteriaRetinanet_Motility_v2.py  --output_dir $low_motility_model_path   --annotations_test "All" --video $video_num   --test_dir $data_path 

      python  src/inferenceBacteriaRetinanet_Motility_v2.py  --output_dir $mid_motility_model_path  --annotations_test "Motility-mid" --video $video_num --test_dir $data_path  
      
      python  src/inferenceBacteriaRetinanet_Motility_v2.py  --output_dir $high_motility_model_path --annotations_test "Motility-high" --video $video_num --test_dir $data_path 
      
      python  src/inferenceBacteriaRetinanet_Motility_v2.py  --output_dir $low_motility_model_path  --annotations_test "Motility-low" --video $video_num --test_dir $data_path  
done