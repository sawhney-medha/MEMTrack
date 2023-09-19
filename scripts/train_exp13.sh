#!/bin/bash
set -e

#test dir
exp_name="collagen_motility_optical_flow_median_bkg_more_data"
data_path="/data/medha/Bacteria/DataFeatures/"
data_path+="$exp_name/data_feature_optical_flow_median_back_2pyr_18win_background_img/"
echo $data_path 


output_dir="/data/medha/Bacteria/training-output/"
output_dir+=$exp_name

 
low_output_dir="$output_dir/low_90k_00125"
high_output_dir="$output_dir/high_90k_00125"
mid_output_dir="$output_dir/mid_90k_00125"
wiggle_output_dir="$output_dir/wiggle_90k_00125"


python Scripts/MotilityAnalysis/trainBacteriaRetinanetMotionData_Motility1.py --source_path $data_path --output_dir $low_output_dir      --annotations_train "Motility-low" --annotations_test "Motility-low" --bbox_size 31 --lr "0.00125" --epochs "90000"    --cuda 4 --exp "low"

# python Scripts/MotilityAnalysis/trainBacteriaRetinanetMotionData_Motility.py --source_path $data_path --output_dir $high_output_dir      --annotations_train "Motility-high" --annotations_test "Motility-high" --bbox_size 31 --lr "0.00125" --epochs "90000"   --cuda 2

# python Scripts/MotilityAnalysis/trainBacteriaRetinanetMotionData_Motility.py --source_path $data_path --output_dir $wiggle_output_dir      --annotations_train "Motility-wiggle" --annotations_test "Motility-wiggle" --bbox_size 31 --lr "0.00125" --epochs "90000"   --cuda 3

# python Scripts/MotilityAnalysis/trainBacteriaRetinanetMotionData_Motility.py --source_path $data_path --output_dir $mid_output_dir      --annotations_train "Motility-mid" --annotations_test "Motility-mid" --bbox_size 31 --lr "0.00125" --epochs "90000"   --cuda 4