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
import os

ap = argparse.ArgumentParser(description='Inference')
ap.add_argument('--video', default='29', type=str, metavar='PATH')
ap.add_argument('--source_path', default="./CombinationModel/data_feature_optical_flow_median_back_2pyr_18win_background_img/", type=str, metavar='PATH')
ap.add_argument('--output_dir', default="CombinationModel/easy-optical_flow_median_back_2pyr_18win_00125_34k", type=str, metavar='CELL PATH')
ap.add_argument('--test_dir', default="CombinationModel/easy-optical_flow_median_back_2pyr_18win_00125_34k", type=str, metavar='CELL PATH')
ap.add_argument('--annotations_train', default="Easy", type=str, metavar='TRAIN')
ap.add_argument('--annotations_test', default="Easy", type=str, metavar='TEST')
ap.add_argument('--epochs', default="90", type=str, metavar='TEST')
ap.add_argument('--lr', default="0.00125", type=str, metavar='TEST')
ap.add_argument('--custom_test_dir', type=str, metavar='CELL PATH')
ap.add_argument('--feature', default="images_feature/", type=str, metavar='FEATURE')

args = ap.parse_args()
print("lr: ", float(args.lr))
coco_format_train = pd.DataFrame(columns=["file_name","height","width","annotations"])
coco_format_train["annotations"] = coco_format_train["annotations"].astype('object')

video_num = args.video

#source = "./try_median_issue_data_feature_optical_flow_median_back_2pyr_18win_background_img/"
source = args.source_path
dest_train = source + "train/"
if args.custom_test_dir:
    dest_test = args.custom_test_dir
else:
    dest_test = args.test_dir + f"/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/test/"
#dest_test = f"./video{video_num}_feature_optical_flow_median_back_2pyr_18win/test/"

val = True 


output_dir = ("try_median_issue_optical_flow_median_back_2pyr_18win_00125_34k_background_img")
#output_dir = ("./optical_flow_median_back/")
output_dir = args.output_dir
train_images_path = "/train" + args.feature #"/images_feature/"
val_image_path = "/val" + args.feature #"/images_feature/"

from contextlib import suppress

with suppress(OSError):
    os.remove(output_dir +'/boardetect_test_coco_format.json.lock')
    os.remove(output_dir +'/boardetect_test_coco_format.json')
    os.remove(source + 'test.json')

#annotations_source = source + "annotation_easy_hard/"
images_source = source + args.feature #"/images_feature/"

if args.annotations_train == "Easy":
    annotations_train = dest_train + "annotation_easy/" 
elif args.annotations_train == "Hard":
    annotations_train = dest_train + "annotation_hard/" 
elif args.annotations_train == "VeryHard":
    annotations_train = dest_train + "annotation_veryhard/" 
elif args.annotations_train == "Easy+Hard":
    annotations_train = dest_test + "annotation_easy_hard/" 
elif args.annotations_train == "All":
    annotations_train = dest_test + "annotation_easy_hard_veryhard/" 
elif args.annotations_train == "Motility-low":
    annotations_train = dest_train + "annotation_motility_low/" 
elif args.annotations_train == "Motility-high":
    annotations_train = dest_train + "annotation_motility_high/" 
elif args.annotations_train == "Motility-wiggle":
    annotations_train = dest_train + "annotation_motility_wiggle/" 
elif args.annotations_train == "Motility-mid":
    annotations_train = dest_train + "annotation_motility_mid/"
elif args.annotations_train == "Motility-motile":
    annotations_train = dest_train + "annotation_motility_wiggle_mid_high/"
elif args.annotations_train == "Sticking-stick":
    annotations_train = dest_train + "annotation_sticking_stick/" 
elif args.annotations_train == "Sticking-motile":
    annotations_train = dest_train + "annotation_sticking_motile/" 
elif args.annotations_train == "Sticking-non_motile":
    annotations_train = dest_train + "annotation_sticking_non_motile/" 
elif args.annotations_train == "Motility-low-wiggle":
    annotations_train = dest_train + "annotation_motility_low_wiggle/" 
elif args.annotations_train == "Motility-mid-high":
    annotations_train = dest_train + "annotation_motility_mid_high/"

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
images_train = dest_train + args.feature #"/images_feature/"
images_test = dest_test + args.feature #"/images_feature/"
test_image_path = images_test

factor_w = 1#1024/1388
factor_h = 1#1024/1040

#function to get background frame
#function to get prev frame
#function to create new image given image num

for txt_file in natsorted(os.listdir(annotations_train)):
        width = 31
        text_file = open(annotations_train + txt_file, 'r')
        xy_coords = text_file.readlines()
        boxes = []
        res=pd.DataFrame(columns=["file_name","height","width","annotations"])
        image = PIL.Image.open(images_train + txt_file[:-4] + ".tif").convert('L')
        image_feature = PIL.Image.open(images_train + txt_file[:-4] + ".tif")
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
        coco_format_train = coco_format_train.append(res)
        coco_format_train.reset_index(drop=True,inplace=True)
        
coco_format_train.reset_index(inplace=True)
coco_format_train.rename(columns={"index":"image_id"},inplace=True)
coco_format_train.to_json(source + "train.json",orient="records")

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
coco_format_test.to_json(source + "test.json",orient="records")



def get_board_dicts(imgdir, mode):
    if mode is 'train':
        json_file = imgdir+"/train.json" #Fetch the json file
    if mode is 'test':
         json_file = imgdir+"/test.json" #Fetch the json file
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        filename = i["file_name"] 
        if mode is 'train':
            i["file_name"] = imgdir + train_images_path + filename 
        if mode is 'test':
             i["file_name"] =  test_image_path + filename 
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYXY_ABS #Setting the required Box Mode
            j["category_id"] = int(j["category_id"])
    return dataset_dicts

#Registering the Dataset
for d in ["train", "test"]:
    DatasetCatalog.register("boardetect_" + d, lambda d=d: get_board_dicts(source, d))
    MetadataCatalog.get("boardetect_" + d).set(thing_classes=["node"])
board_metadata = MetadataCatalog.get("boardetect_train")
val_metadata = MetadataCatalog.get("boardetect_test")

train_data = ("boardetect_train",)
test_data = ("boardetect_test",)
if val ==True:
    val_data = ("boardetect_test",)
else:
    val_data = ("boardetect_train",)



class CocoTrainer(DefaultTrainer):
       
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            output_folder = cfg.OUTPUT_DIR


        return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cuda:0' 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
#Passing the Train and Validation sets
cfg.DATASETS.TRAIN = train_data #("boardetect_train",)
cfg.DATASETS.TEST = test_data #("boardetect_train",)
cfg.OUTPUT_DIR = output_dir #("comparison-optical-flow")
# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #uncommwnt during inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #uncommwnt during inference
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0029999.pth") #uncommwnt during inference
# Number of images per batch across all machines.
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = float(args.lr) #0.0125  # pick a good LearningRate
cfg.SOLVER.MAX_ITER = int(args.epochs) #30000  #No. of iterations
# cfg.SOLVER.STEPS = (300,600)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 80
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.TEST.EVAL_PERIOD = 500 # No. of iterations after which the Validation Set is evaluated. 
cfg.TEST.DETECTIONS_PER_IMAGE = 60
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.VIS_PERIOD = 500
cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 100
cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 100
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 80 #80
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 80
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg) 
trainer.resume_or_load(resume=True) #resume=True for inference
trainer.train()
print(cfg.MODEL.WEIGHTS)

os.system(f'cp {output_dir}/coco_instances_results.json {dest_test}/coco_instances_results_{args.annotations_test}.json')  
os.system(f'cp {output_dir}/boardetect_test_coco_format.json {dest_test}/boardetect_test_coco_format_{args.annotations_test}.json')  
os.system(f'cp {source}/test.json {dest_test}/test_{args.annotations_test}.json')  