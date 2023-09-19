#!/bin/bash
set -e
exp_name="collagen_motility_optical_flow_median_bkg_more_data" # exp name
data_path="/data/medha/Bacteria/DataFeatures/" # feature directory path
data_path+=$exp_name
echo $data_path 

source_path="/data/medha/Bacteria/DataFeatures/" # feature directory path
source_path+=$exp_name
source_path+="/data_feature_optical_flow_median_back_2pyr_18win_background_img/" 
echo $source_path

#path to saved models
low_motility_model_path="./Models/motility/low/collagen_optical_flow_median_bkg_more_data_90k/" 
wiggle_motility_model_path="./Models/motility/wiggle/collagen_optical_flow_median_bkg_more_data_90k/" 
mid_motility_model_path="./Models/motility/mid/collagen_optical_flow_median_bkg_more_data_90k/" 
high_motility_model_path="./Models/motility/high/collagen_optical_flow_median_bkg_more_data_90k/"  


#update test video numbers from video map
for video_num in  141 148 153 160 167 170  
do
   
   #To genearate testing files for all motilities
      python Scripts/MotilityAnalysis/inferenceBacteriaRetinanet_Motility.py --source_path $source_path --output_dir $low_motility_model_path --annotations_train "All" --annotations_test "All" --video $video_num  --lr "0.00125" --epochs "90000" --test_dir $data_path 

      python Scripts/MotilityAnalysis/inferenceBacteriaRetinanet_Motility.py --source_path $source_path --output_dir $mid_motility_model_path --annotations_train "Motility-mid" --annotations_test "Motility-mid" --video $video_num --test_dir $data_path  --lr "0.00125" --epochs "90000"

      python Scripts/MotilityAnalysis/inferenceBacteriaRetinanet_Motility.py --source_path $source_path --output_dir $high_motility_model_path --annotations_train "Motility-high" --annotations_test "Motility-high" --video $video_num --test_dir $data_path  --lr "0.00125" --epochs "90000"
      
     python Scripts/MotilityAnalysis/inferenceBacteriaRetinanet_Motility.py --source_path $source_path --output_dir $low_motility_model_path --annotations_train "Motility-low" --annotations_test "Motility-low" --video $video_num --test_dir $data_path  --lr "0.00125" --epochs "90000"  
done
