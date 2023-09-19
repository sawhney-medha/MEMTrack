#!/bin/bash
set -e

exp_name="collagen_motility_optical_flow_median_bkg_more_data/" # exp name
source_path="/data/medha/Bacteria" 
data_path="$source_path/DataFeatures/$exp_name" # path toi feature directory
echo $data_path 

video_map_path="$source_path/Data/videomap.txt" # path to video map
echo $video_map_path

test_results_path="$data_path/test_set_results*.txt" #path to test result output files

rm -rf $test_results_path

for video_num in  141 148 153 160 167 170  
do  
     python Scripts/MotilityAnalysis/experiments/analysis/evaluation_step_wise_motility.py --video_map_path $video_map_path --data_path $data_path --video $video_num
done
