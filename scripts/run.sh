#!/bin/bash

set -e 

#RUN ONLY INFERENCE WITH EVAL(GT DATA)
bash scripts/inference.sh 
#run with plot and plt gt
bash scripts/tracking.sh 
bash scripts/test_set_eval.sh

# RUN ONLY INFERENCE NO EVAL(NO GT DATA)
 bash scripts/inference_no_eval.sh
# #run only with plot
# bash scripts/tracking.sh 
