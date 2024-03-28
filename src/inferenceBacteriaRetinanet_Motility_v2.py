# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2
import pandas as pd
import shutil
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import argparse
from natsort import natsorted
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
import os

ap = argparse.ArgumentParser(description='Inference')
ap.add_argument('--video', default='29', type=str, metavar='PATH')
ap.add_argument('--output_dir', default="CombinationModel/easy-optical_flow_median_back_2pyr_18win_00125_34k", type=str, metavar='CELL PATH')
ap.add_argument('--test_dir', default="CombinationModel/easy-optical_flow_median_back_2pyr_18win_00125_34k", type=str, metavar='CELL PATH')
ap.add_argument('--annotations_test', default="Easy", type=str, metavar='TEST')
ap.add_argument('--custom_test_dir', type=str, metavar='CELL PATH')
ap.add_argument('--feature', default="images_feature/", type=str, metavar='FEATURE')

args = ap.parse_args()

video_num = args.video

if args.custom_test_dir:
    dest_test = args.custom_test_dir
else:
    dest_test = args.test_dir + f"/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/"
#dest_test = f"./video{video_num}_feature_optical_flow_median_back_2pyr_18win/test/"



output_dir = ("try_median_issue_optical_flow_median_back_2pyr_18win_00125_34k_background_img")
#output_dir = ("./optical_flow_median_back/")
output_dir = args.output_dir
val_image_path = "/val" + args.feature #"/images_feature/"

from contextlib import suppress

with suppress(OSError):
    os.remove(output_dir +'/boardetect_test_coco_format.json.lock')
    os.remove(output_dir +'/boardetect_test_coco_format.json')
    os.remove(dest_test + 'test.json')



if args.annotations_test == "Easy":
    annotations_test = dest_test + "annotation_easy/" 
elif args.annotations_test == "Hard":
    annotations_test = dest_test + "annotation_hard/" 
elif args.annotations_test == "VeryHard":
    annotations_test = dest_test + "annotation_veryhard/" 
elif args.annotations_test == "Easy+Hard":
    annotations_test = dest_test + "annotation_easy_hard/" 
elif args.annotations_test == "All":
    annotations_test = dest_test + "annotation_easy_hard_veryhard/" 
elif args.annotations_test == "Motility-low":
    annotations_test = dest_test + "annotation_motility_low/" 
elif args.annotations_test == "Motility-high":
    annotations_test = dest_test + "annotation_motility_high/" 
elif args.annotations_test == "Motility-wiggle":
    annotations_test = dest_test + "annotation_motility_wiggle/" 
elif args.annotations_test == "Motility-mid":
    annotations_test = dest_test + "annotation_motility_mid/" 
elif args.annotations_test == "Motility-motile":
    annotations_test = dest_test + "annotation_motility_wiggle_mid_high/"
elif args.annotations_test == "Sticking-stick":
    annotations_test = dest_test + "annotation_sticking_stick/"
elif args.annotations_test == "Sticking-motile":
    annotations_test = dest_test + "annotation_sticking_motile/" 
elif args.annotations_test == "Sticking-non_motile":
    annotations_test = dest_test + "annotation_sticking_non_motile/" 
elif args.annotations_test == "Motility-low-wiggle":
    annotations_test = dest_test + "annotation_motility_low_wiggle/" 
elif args.annotations_test == "Motility-mid-high":
    annotations_test = dest_test + "annotation_motility_mid_high/" 



#To test a particular bacteria
#annotations_test = dest_test + "bacteria/4/xy_coord/"
images_test = dest_test + args.feature #"/images_feature/"
test_image_path = images_test

factor_w = 1#1024/1388
factor_h = 1#1024/1040

#function to get background frame
#function to get prev frame
#function to create new image given image num

coco_format_test = pd.DataFrame(columns=["file_name","height","width","annotations"])
coco_format_test["annotations"] = coco_format_test["annotations"].astype('object')

for txt_file in natsorted(os.listdir(annotations_test)):
        width = 31
        text_file = open(annotations_test + txt_file, 'r')
        xy_coords = text_file.readlines()
        boxes = []
        res=pd.DataFrame(columns=["file_name","height","width","annotations"])
        image = PIL.Image.open(images_test + txt_file[:-4] + ".tif").convert('L')
        image_feature = PIL.Image.open(images_test + txt_file[:-4] + ".tif")
        image = image_feature
        #print(image.size)
        res.at[0,"height"] = image.height
        res.at[0,"width"] = image.width
        res.at[0,"file_name"] = txt_file[:-4]+".tif"
        bbox_mode = 0
        category_id = 0
        # image2 = image.resize((1024,1024))
        # image2.save(images_resized_train + txt_file[:-4] + ".jpg")
        for xy in xy_coords:
            box = []
            x = float(xy.split(" ")[0])
            y = float(xy.split(" ")[1])
            x1 = int(x*factor_w - (width // 2))
            y1 = int(y*factor_h - (width // 2))
            x2 = int(x*factor_w + (width // 2))
            y2 = int(y*factor_h + (width // 2))
            w = h = 31
            box = [x1, y1, x2, y2]
            boxes.append(np.array(box))
            #print(np.array(box))

        res["annotations"]=res["annotations"].astype('object')
        annotation_df = pd.DataFrame(columns=["bbox","bbox_mode","category_id"])
        annotation_df["bbox"] = boxes
        annotation_df["bbox_mode"] = bbox_mode
        annotation_df["category_id"] = category_id
        annotations = annotation_df.T.to_dict().values()
        l = []
        for j in annotations:
            l.append(j)
        res.at[0,"annotations"] = l
        coco_format_test = coco_format_test.append(res)
        coco_format_test.reset_index(drop=True,inplace=True)
        
coco_format_test.reset_index(inplace=True)
coco_format_test.rename(columns={"index":"image_id"},inplace=True)
coco_format_test.to_json(dest_test + "test.json",orient="records")



def get_board_dicts(imgdir, mode):
    if mode == 'test':
         json_file = imgdir+"/test.json" #Fetch the json file
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        filename = i["file_name"] 
        if mode == 'test':
             i["file_name"] =  test_image_path + filename 
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYXY_ABS #Setting the required Box Mode
            j["category_id"] = int(j["category_id"])
    return dataset_dicts

#Registering the Dataset
for d in ["test"]:
    DatasetCatalog.register("boardetect_" + d, lambda d=d: get_board_dicts(dest_test, d))
    MetadataCatalog.get("boardetect_" + d).set(thing_classes=["node"])

test_metadata = MetadataCatalog.get("boardetect_test")

test_data = ("boardetect_test",)

    
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cuda:0' 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
cfg.DATASETS.TEST = test_data #("boardetect_train",)
cfg.OUTPUT_DIR = output_dir #("comparison-optical-flow")
# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0089999.pth") #uncommwnt during inference
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0029999.pth") #uncommwnt during inference
# Number of images per batch across all machines.
cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.SOLVER.STEPS = (300,600)
cfg.TEST.DETECTIONS_PER_IMAGE = 60
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(cfg.MODEL.WEIGHTS)

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("boardetect_test", cfg, False, output_dir)
val_loader = build_detection_test_loader(cfg, "boardetect_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

os.system(f'cp {output_dir}/coco_instances_results.json {dest_test}/coco_instances_results_{args.annotations_test}.json')  
os.system(f'cp {output_dir}/boardetect_test_coco_format.json {dest_test}/boardetect_test_coco_format_{args.annotations_test}.json')  
os.system(f'cp {dest_test}/test.json {dest_test}/test_{args.annotations_test}.json')