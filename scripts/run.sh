#!/bin/bash

set -e 

#RUN ONLY INFERENCE WITH EVAL(GT DATA)
#bash Scripts/detection/inference_eval.sh 
#run with plot and plt gt
bash Scripts/tracking/tracking.sh 
bash Scripts/analysis/test_set_eval.sh

# #RUN ONLY INFERENCE NO EVAL(NO GT DATA)
# bash Scripts/detection/inference_no_eval.sh
# #run only with plot
# bash Scripts/tracking/tracking.sh 


# #ONLY INFERENCE SINGLE CATEGORY LABELS
# bash Scripts/samples/inference_label_singlecategory.sh
# bash Scripts/tracking/tracking.sh
# bash Scripts/analysis/test_set_eval.sh