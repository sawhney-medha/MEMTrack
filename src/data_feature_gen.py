import zipfile
import openpyxl
import os
import tqdm
import csv
import cv2
import shutil
import PIL
import glob
import pandas as pd
import numpy as np
from natsort import natsorted
from PIL import Image
import argparse


def create_video(data_dir):
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    #print(data_dir)
    img_sample = cv2.imread(os.path.join(data_dir,"images/0.tif"))
    #print(img_sample.shape)
    height, width, channels = img_sample.shape
    
    video = cv2.VideoWriter(data_dir + 'video.mp4', fourcc, 1, (width, height))
    #data_dir = "./Data/video3/"
    image_dir = os.path.join(data_dir, "images")
    for frame in natsorted(os.listdir(image_dir)):
        #print(frame)
        img = cv2.imread(os.path.join(image_dir, frame))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    
    
    
def get_background(file_path, mean=True, sample=False):
    if not os.path.exists(file_path):
        create_video(file_path.rsplit("/",1)[0] +"/frame1")
    cap = cv2.VideoCapture(file_path)
    #print(cap.read())
    # we will randomly select 50 frames for the calculating the median
    #frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=20)
    frame_indices = list(range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) ))
    print(len(frame_indices))
    # we will store the frames in array
    frames = []
    for idx in frame_indices:
        # set the frame id to read that particular frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        #print(ret)
        frames.append(frame)
    if mean:
         # calculate the mean
        background_frame = np.mean(frames, axis=0).astype(np.uint8)
    else:
        # calculate the median
        background_frame = np.median(frames, axis=0).astype(np.uint8)
    if sample==True:
        background_frame = cv2.imread("./Control_2_b0t5306c0x0-660y0-492.tiff")
        #background_frame = cv2.imread("./RBS 2_1_b0t2791c0x0-660y0-492.tiff")
    return background_frame


def get_absolute_consec_frame_diff_feature(video_path, max_consec_frame_diff=True, num_prior_frames=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    differences = []
    prev_frames = []
    while True:
        ret, frame = cap.read()
        #print(frame)
        if not ret:
            break
        #storing old frames till prior frame number
        if num_prior_frames>0:
            prev_frames.append(frame)
            num_prior_frames -= 1
            continue
        #retrieving the previous xth frame
        prev = prev_frames[0]
        prev_frames.pop(0)
        #computing frame diff between current and one previous
        consecutive_diff = np.abs(frame - prev)
        frames.append(frame) #creating frame list
        #creating consecutive frame diff list
        differences.append(consecutive_diff)
    #creating consecutive frame diff features by taking the max at every pixel along the frame diff list
    if max_consec_frame_diff:
        max_abs_consec_diff_feature = np.max(differences, axis=0)
    else:
         max_abs_consec_diff_feature = np.min(differences, axis=0)
    return max_abs_consec_diff_feature



def get_diff_from_absolute_consec_frame_diff_feature(image_path, frame_diff_feature, frame=False):
    if frame==False:
        image = PIL.Image.open(image_path).convert('L')
    if frame == True:
        #print("getting diff from frame")
        image = Image.fromarray(image_path).convert('L')
    L = image.getchannel(0)
    frame_diff_feature = PIL.Image.fromarray(frame_diff_feature).convert('L')
    frame_diff_feature.getbands()
    L1 = frame_diff_feature.getchannel(0)
    diff = PIL.ImageChops.difference(L, L1)
    return diff


def get_absolute_all_frame_diff_feature(video_path, max_feature=True):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        #print(frame)
        if not ret:
            break
        frames.append(frame)
    
    features = []
    count = 0
    for index in tqdm.tqdm(range(len(frames))):
        differences = []
        count+=1
        frame = frames[index]
        for index_1 in (range(len(frames))):
            if index == index_1:
                #print(count)
                continue
            frame1 = frames[index_1]
            differences.append(np.abs(frame - frame1))
        if max_feature:
            max_diff_feature = np.max(differences, axis=0)
            features.append(max_diff_feature)
        else:
            #print("min")
            min_diff_feature = np.min(differences, axis=0)
            features.append(min_diff_feature)
    return features    



#updated to include optical floiw computtaion from x frames prior
def gen_dense_optical_flow_data(method, video_path, params=[], to_gray=False, median=False, median_frame=None, num_frames_prior=1):
    frames_optical_flow = []
    frames_orignal = []
    # Read the video and first x frames
    cap = cv2.VideoCapture(video_path)
    #print("fps",cap.get(cv2.CAP_PROP_FPS))
    old_frames = []
    for i in range(num_frames_prior):
        ret, old_frame = cap.read()
        # crate HSV & make Value a constant
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255
        if to_gray:
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_frames.append(old_frame)

    #to compute optical flow from the median background    
    if median == True:
        old_frame = median_frame
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_GRAY2BGR)
        
  

    
    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
        
        old_frame = old_frames[0]
        if median == True:
            old_frame = median_frame

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                
            
        # Calculate Optical Flow
      
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #plt.imshow(frame_copy)
        #plt.imshow(bgr)
        # plt.imshow(v,cmap='gray', vmin=0, vmax=255)
        # plt.show()
    
        # Update the previous frame
        old_frames.append(new_frame)
        old_frames.pop(0)
        frames_orignal.append(frame_copy)
        frames_optical_flow.append(bgr)
    return frames_orignal, frames_optical_flow



def get_background_diff_feature(image_path, background):
    image = PIL.Image.open(image_path).convert('L')
    L = image.getchannel(0)
    background_img = PIL.Image.fromarray(background)
    background_img.getbands()
    L1 = background_img.getchannel(0)
    diff = PIL.ImageChops.difference(L, L1)
    return diff
    
def get_prev_frame_diff_feature(image_path, prev_image_path):
    image = PIL.Image.open(image_path).convert('L')
    L = image.getchannel(0)
    image = PIL.Image.open(prev_image_path).convert('L')
    L1 = image.getchannel(0)
    diff = PIL.ImageChops.difference(L, L1)
    return diff

def create_feature_image(image_path, background, prev_image_path, prev_image_path2=None):
    prev_image_diff = get_prev_frame_diff_feature(image_path, prev_image_path)
    image = PIL.Image.open(image_path).convert('L')
    L = image.getchannel(0)
    if prev_image_path2 is None:
        background_diff = get_background_diff_feature(image_path, background)
        newImagediff = PIL.Image.merge("RGB", [L, background_diff, prev_image_diff])
    if prev_image_path2 is not None:
        prev_image_diff2 = get_prev_frame_diff_feature(image_path, prev_image_path2)
        newImagediff = PIL.Image.merge("RGB", [L, prev_image_diff2, prev_image_diff])
        
    return newImagediff



def create_feature_image_optical_flow(frame, optical_flow, pure=False, background=None, optical_flow2=None, final_channel=False):
    frame = Image.fromarray(frame).convert('L')
    L = frame.getchannel(0)
    flow = PIL.Image.fromarray(optical_flow)
    #flow.save("flow.png")
    hsv_optical_flow = cv2.cvtColor(optical_flow, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_optical_flow)
    flow = PIL.Image.fromarray(v)
    #flow.save("flow_hsv.png")
    
    #print(v.shape)
    last_channel = np.zeros([v.shape[0], v.shape[1]],dtype=np.uint8)
    last_channel[:] = 255
    v = Image.fromarray(v).convert('L')
    v = v.getchannel(0)
    
    
    #print(last_channel)
    last_channel = Image.fromarray(last_channel).convert('L')
    last_channel = last_channel.getchannel(0)

    gray_optical_flow = Image.fromarray(cv2.cvtColor(optical_flow, cv2.COLOR_BGR2GRAY)).convert('L').getchannel(0)
    
    feature_image = PIL.Image.merge("RGB", [L, v, last_channel])
    feature_image = PIL.Image.merge("RGB", [L, gray_optical_flow, last_channel])
    
    if pure == False:
        if background is None:
            gray_optical_flow2 = Image.fromarray(cv2.cvtColor(optical_flow2, cv2.COLOR_BGR2GRAY)).convert('L').getchannel(0)
            feature_image = PIL.Image.merge("RGB", [L, gray_optical_flow, gray_optical_flow2])
            #print(here)
        else:
            if final_channel == False:
                background_img = PIL.Image.fromarray(background)
                L1 = background_img.getchannel(0)
                feature_image = PIL.Image.merge("RGB", [L, v, L1]) #-->org code
                # diff = PIL.ImageChops.difference(L, L1)
                # feature_image = PIL.Image.merge("RGB", [L, v, diff]) #-->up code
                #print("here adding background")
                #print("background")
            if final_channel==True:
                #print("final channel tru, just adding diff")
                feature_image = PIL.Image.merge("RGB", [L, v, background]) 
            
       
        
    # print(feature_image.size)
    # print(feature_image.mode)
        
    return feature_image



def save_data(image_dest, count, feature_image, frame_num, data_dir, video, minivideo, count_test, count_train, train):
    #save image feature
    feature_image.save(image_dest +"images_feature/"+str(count)+".tif")

    text_file = str(frame_num) +".txt"
    annotations_easy_source = os.path.join(data_dir, video, minivideo, "annotations_easy",
                                           text_file)
    annotations_easy_hard_source = os.path.join(data_dir, video, minivideo, 
                                            "annotations_easy_hard", text_file)
    
    annotations_very_hard_source = os.path.join(data_dir, video, minivideo, "annotations_veryhard",
                                           text_file)
    annotations_easy_hard_veryhard_source = os.path.join(data_dir, video, minivideo, 
                                            "annotations_easy_hard_veryhard", text_file)


    shutil.copy(annotations_easy_source, image_dest +"annotation_easy/" +str(count)+".txt")
    shutil.copy(annotations_easy_hard_source, image_dest +"annotation_easy_hard/" 
                + str(count)+".txt")
    if os.path.exists(annotations_very_hard_source):
        shutil.copy(annotations_very_hard_source, image_dest +"annotation_veryhard/" +str(count)+".txt")
    if os.path.exists(annotations_easy_hard_veryhard_source):
        shutil.copy(annotations_easy_hard_veryhard_source, image_dest +"annotation_easy_hard_veryhard/" 
                + str(count)+".txt")
    #create hard only annotation
    text_file_easy_hard = open(annotations_easy_hard_source, 'r')
    xy_coords_easy_hard  = text_file_easy_hard.readlines()

    text_file_easy = open(annotations_easy_source, 'r')
    xy_coords_easy  = text_file_easy.readlines()

    xy_coords_hard = [coord for coord in xy_coords_easy_hard if coord not in xy_coords_easy ] 
    text_file_hard = open(image_dest +"annotation_hard/" +str(count)+".txt", 'w')
    for coord in xy_coords_hard:
        text_file_hard.write(coord)
    text_file_hard.close()

    annotations_low_motility_source = os.path.join(data_dir, video, minivideo, "annotations_motility_low",
                                           text_file)
    annotations_high_motility_source = os.path.join(data_dir, video, minivideo, "annotations_motility_high",
                                           text_file)
    annotations_mid_motility_source = os.path.join(data_dir, video, minivideo, "annotations_motility_mid",
                                           text_file)
    annotations_wiggle_motility_source = os.path.join(data_dir, video, minivideo, "annotations_motility_wiggle",
                                           text_file)
    
    annotations_sticking_motile_source = os.path.join(data_dir, video, minivideo, "annotations_sticking_motile",
                                           text_file)
    annotations_sticking_stick_source = os.path.join(data_dir, video, minivideo, "annotations_sticking_stick",
                                           text_file)
    annotations_sticking_non_motile_source = os.path.join(data_dir, video, minivideo, "annotations_sticking_non_motile",
                                           text_file)
    
    shutil.copy(annotations_low_motility_source, image_dest +"annotation_motility_low/" +str(count)+".txt")
    shutil.copy(annotations_high_motility_source, image_dest +"annotation_motility_high/" +str(count)+".txt")
    shutil.copy(annotations_mid_motility_source, image_dest +"annotation_motility_mid/" +str(count)+".txt")
    shutil.copy(annotations_wiggle_motility_source, image_dest +"annotation_motility_wiggle/" +str(count)+".txt")
    
    shutil.copy(annotations_sticking_stick_source, image_dest +"annotation_sticking_stick/" +str(count)+".txt")
    shutil.copy(annotations_sticking_motile_source, image_dest +"annotation_sticking_motile/" +str(count)+".txt")
    shutil.copy(annotations_sticking_non_motile_source, image_dest +"annotation_sticking_non_motile/" +str(count)+".txt")
    
    
    if train == True:
        count_train += 1
    else:
        count_test += 1
        
    return count_test, count_train
    
    
    
#get backgorund frame for every mini video
#skip the first frame in every mini video
#store image in train images set
#store image+background diff + prev image diff in train images feature set
#similarly for test

def create_data(data_dir, dest_dir, trainfolder, train_video, testfolder, test_video, valfolder, val_video, method="background", num_prev=None, mean=False, sample=False, test_only = False, params=None, optical_flow_prior=1, frame_diff_prior=1):
    os.makedirs(dest_dir, exist_ok=True)
    data_dir_types = ["/images/", "/images_feature/", "/annotation_easy/", "/annotation_hard/", "/annotation_easy_hard/", "/annotation_easy_hard_veryhard/" , "/annotation_veryhard/" , "/annotation_motility_low", "/annotation_motility_wiggle", "/annotation_motility_mid", "/annotation_motility_high", "/annotation_sticking_stick", "/annotation_sticking_motile", "/annotation_sticking_non_motile"]
    for video in test_video:
        for dir_type in data_dir_types:
            os.makedirs(os.path.join(dest_dir, testfolder, video) + dir_type, exist_ok=True)
            
    for video in val_video:
        for dir_type in data_dir_types:
            if test_only == True:
                continue
            os.makedirs(os.path.join(dest_dir, valfolder, video) + dir_type, exist_ok=True)
            
    for dir_type in data_dir_types:
            os.makedirs(os.path.join(dest_dir, testfolder) + dir_type, exist_ok=True)
            if test_only == True:
                continue
            os.makedirs(os.path.join(dest_dir, valfolder) + dir_type, exist_ok=True)
            os.makedirs(dest_dir+trainfolder + dir_type, exist_ok=True)
    
    count_train = 0
    count_test = 0
    count_test_all = 0
    count_val_all = 0

    for video in natsorted(os.listdir(data_dir)):
        
        if not video.startswith('.') and os.path.isdir(os.path.join(data_dir,video))==True:
            if test_only == True and video not in test_video:
                continue
            if video not in train_video + test_video + val_video:
                continue
            for minivideo in natsorted(os.listdir(os.path.join(data_dir,video))) :
                if not minivideo.startswith('.') and os.path.isdir(os.path.join(data_dir,video,minivideo)) == True:
                    video_path = os.path.join(data_dir,video) + "/" + minivideo + "video.mp4"
                    print(video_path)

                    if method == "background":
                        # get the background model
                        background = get_background(video_path, mean, sample=sample)
                        # convert the background model to grayscale format
                        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                        skip_frame_list = [0]
                    if method == "nprev":
                        skip_frame_list = list(range(num_prev))
                    if method in ["diff_from_max_absolute_consecutive_frame_diff", "max_absolute_consecutive_frame_diff"]:
                        max_absolute_consecutive_frame_diff = get_absolute_consec_frame_diff_feature(video_path, max_consec_frame_diff=True)
                    if method in ["max_absolute_all_frame_diff"]:
                        absolute_all_frame_diff = get_absolute_all_frame_diff_feature(video_path, max_feature=True)
                    if method in ["min_absolute_all_frame_diff"]:
                        absolute_all_frame_diff = get_absolute_all_frame_diff_feature(video_path, max_feature=False)
                        
                        
                        
                    if video not in (test_video + val_video): 
                        #print(video)
                        train=True
                    else:
                        train=False
                        count_test = 0
                        if video in test_video:
                            testfolder_video = os.path.join(testfolder, video)
                        else:
                            testfolder_video = os.path.join(valfolder, video)
                    
                        
                    if method in ["optical_flow", "optical_flow_median_back", "optical_flow_from_median_frame", 
                                  "optical_flow_combined", "diff_from_max_absolute_consecutive_frame_diff", 
                                  "max_absolute_consecutive_frame_diff", "min_absolute_all_frame_diff", 
                                  "max_absolute_all_frame_diff"]:
                        #print(method)
                        method_flow = cv2.calcOpticalFlowFarneback
                        #params = [0.5, 3, 15, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters
                        # params = [0.5, 4, 18, 3, 5, 1.2, 0]  
                        #params = [0.5, 2, 18, 3, 5, 1.2, 0]  
                        #params = [0.5, 2, 20, 3, 5, 1.2, 0]  
                        
                        background = get_background(video_path, mean, sample=sample)
                        # convert the background model to grayscale format
                        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                        
                        if method == "optical_flow_from_median_frame":
                            frames_org, frames_optical_flow = gen_dense_optical_flow_data(method_flow, video_path, params, to_gray=True, median=True, median_frame=background)
                        else :
                            frames_org, frames_optical_flow = gen_dense_optical_flow_data(method_flow, video_path, params, to_gray=True,num_frames_prior=optical_flow_prior)
                            
                        if method == "optical_flow_combined":
                            m_frames_org, m_frames_optical_flow = gen_dense_optical_flow_data(method_flow, video_path, params, to_gray=True, median=True, median_frame=background)
                            c_frames_org, c_frames_optical_flow = gen_dense_optical_flow_data(method_flow, video_path, params, to_gray=True)
    #                     print(len(frames_org))
    #                     print(frames_org[0].shape)
    #                     print(len(frames_optical_flow))
    #                     print(frames_optical_flow[0].shape)
    
                        if method in ["optical_flow", "optical_flow_from_median_frame"]:
                            pure = True
                        else:
                            pure = False
                                              

                        print(len(frames_org))        
                        for i, frame in enumerate(frames_org):
                            #save frame in images
                            if train==True:
                                image_dest = dest_dir + "/" +trainfolder +"/"
                                count = count_train
                            else:
                                #print(video)
                                image_dest = dest_dir +"/" + testfolder_video + "/"
                                count = count_test

                            #print(frame.shape)
                            img = Image.fromarray(frame, "RGB")
                            img.save(image_dest +"images/"+str(count)+".tif")
                            if method == "optical_flow_combined":
                                feature_image = create_feature_image_optical_flow(frame, c_frames_optical_flow[i], pure=pure, optical_flow2=m_frames_optical_flow[i])
                            elif method == "diff_from_max_absolute_consecutive_frame_diff":
                                diff_from_max_absolute_consecutive_frame_diff = get_diff_from_absolute_consec_frame_diff_feature(frame_diff_feature=max_absolute_consecutive_frame_diff, image_path=frame, frame=True)
                                feature_image = create_feature_image_optical_flow(frame, frames_optical_flow[i], pure, diff_from_max_absolute_consecutive_frame_diff, final_channel=True)
                            elif method == "max_absolute_consecutive_frame_diff":
                                feature_image = create_feature_image_optical_flow(frame, frames_optical_flow[i], pure, max_absolute_consecutive_frame_diff, final_channel=False)
                            elif method in ["max_absolute_all_frame_diff", "min_absolute_all_frame_diff"]:
                                feature_image = create_feature_image_optical_flow(frame, frames_optical_flow[i], pure, absolute_all_frame_diff[i+1], final_channel=False)
                            else:
                                feature_image = create_feature_image_optical_flow(frame, frames_optical_flow[i], pure, background)
                            frame_num = i+1
                            
                            count_test, count_train = save_data(image_dest, count, feature_image, frame_num, data_dir, video, minivideo, count_test, 
                                                                                count_train, train)
                            if train == False:
                                if video in test_video:
                                    count = count_test_all
                                    image_dest_all = dest_dir +"/" + testfolder + "/"
                                    count_test_all, count_train = save_data(image_dest_all, count, feature_image, frame_num, data_dir, video, minivideo, count_test_all, 
                                                                                 count_train, train)
                                else:
                                    count = count_val_all
                                    image_dest_all = dest_dir +"/" + valfolder + "/"
                                    count_val_all, count_train = save_data(image_dest_all, count, feature_image, frame_num, data_dir, video, minivideo, count_val_all, 
                                                                                 count_train, train)
                                img.save(image_dest_all +"images/"+str(count)+".tif")
                                
                            
                    else:
                        # print("in else")
                        # print(method)
                        for frame in natsorted(os.listdir(os.path.join(data_dir, video, minivideo, "images"))):
                            frame_num = int(frame.split(".tif")[0])
                            #skip first frame
                            if frame_num not in skip_frame_list:
                                #save frame in images
                                images_source = os.path.join(data_dir, video, minivideo, "images", frame)
                                if train==True:
                                    image_dest = dest_dir + "/" +trainfolder +"/"
                                    count = count_train
                                    #print(count)
                                else:
                                    #print(video)
                                    image_dest = dest_dir +"/" + testfolder_video + "/"
                                    image_dest_all = dest_dir +"/" + testfolder + "/"
                                    count = count_test

                                shutil.copy(images_source, image_dest +"images/"+str(count)+".tif")

                                #create new image
                                prev_frame = str(frame_num-1) +".tif"
                                prev_image = os.path.join(data_dir, video, minivideo, "images", prev_frame) 
                                if method == "background":
                                    feature_image = create_feature_image(image_path=images_source, background=background,prev_image_path=prev_image)
                                if method == "nprev":
                                    background=None
                                    prev_frame2 = str(frame_num-num_prev) +".tif"
                                    prev_image2 = os.path.join(data_dir, video, minivideo, "images", prev_frame2) 
                                    feature_image = create_feature_image(image_path=images_source, background=background, 
                                                                     prev_image_path=prev_image, prev_image_path2=prev_image2)



                                count_test, count_train = save_data(image_dest, count, feature_image, frame_num, data_dir, video, minivideo, count_test, count_train, train)
                                if train == False:
                                    if video in test_video:
                                        count = count_test_all
                                        image_dest_all = dest_dir +"/" + testfolder + "/"
                                        count_test_all, count_train = save_data(image_dest_all, count, feature_image, frame_num, data_dir, video, minivideo, count_test_all, 
                                                                                     count_train, train)
                                    else:
                                        count = count_val_all
                                        image_dest_all = dest_dir +"/" + valfolder + "/"
                                        count_val_all, count_train = save_data(image_dest_all, count, feature_image, frame_num, data_dir, video, minivideo, count_val_all, 
                                                                                     count_train, train)
                                    shutil.copy(images_source, image_dest_all +"images/"+str(count)+".tif")
                                    
                            

def create_train_data(target_data_sub_dir, dest_sub_dir, exp_name, train_video,  test_video, val_video, feature_method="optical_flow_median_back"):
    feature_dir = "data_feature_optical_flow_median_back_2pyr_18win_background_img/"
    data_dir = target_data_sub_dir
    dest_dir = os.path.join(dest_sub_dir, exp_name, feature_dir)
    trainfolder = "train"
    testfolder = "test"
    valfolder = "val"
    params = [0.5, 2, 18, 3, 5, 1.2, 0] 

    create_data(data_dir, dest_dir, trainfolder, train_video, testfolder, test_video, valfolder, val_video, 
            method=feature_method, params=params)

    
    
def create_test_data(target_data_sub_dir, dest_sub_dir, exp_name, test_video_list, feature_method="optical_flow_median_back"):
    params = [0.5, 2, 18, 3, 5, 1.2, 0] 
    for video in test_video_list:
        print(video)
        data_dir = target_data_sub_dir
        dest_dir = os.path.join(dest_sub_dir, exp_name , f"data_{video}_feature_optical_flow_median_back_2pyr_18win_background_img/")
        trainfolder = "train"
        testfolder = "test"
        valfolder = "val"
        val_video = []
        train_video = []
        test_video = [f"{video}"]
        params = [0.5, 2, 18, 3, 5, 1.2, 0]  

        create_data(data_dir, dest_dir, trainfolder, train_video, testfolder, test_video, valfolder, val_video, 
                    method=feature_method, test_only = True, params=params)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature Preparation")
    parser.add_argument("--target_data_sub_dir", default = "MEMTrack/data/collagen/", help="Path to the folder to process")
    parser.add_argument("--dest_sub_dir",  default ="MEMTrack/DataFeatures/", help="Path to the folder to process")
    parser.add_argument("--exp_name", default ="collagen_motility_inference",  help="Path to the folder to process")
    parser.add_argument("--feature_method",  default="optical_flow_median_back", help="Path to the folder to process")
    # optical flow median back with optical_Flow_prior is optical flow from xth previous frame
    # "diff_from_max_absolute_consecutive_frame_diff" creates a feature for diff from the "max consecutive frame diff" feature, with a frame diff prior for xth frame diff
    parser.add_argument('--train_video', type=str, nargs='+', help='a list of strings', default=[])
    parser.add_argument('--val_video',type=str, nargs='+', help='a list of strings', default=[])
    parser.add_argument('--test_video', type=str, nargs='+', help='a list of strings', default=[])

    args = parser.parse_args(args)
    target_data_sub_dir = args.target_data_sub_dir
    dest_sub_dir  = args.dest_sub_dir
    exp_name = args.exp_name
    feature_method = args.feature_method
    train_video = args.train_video
    test_video = args.test_video
    val_video = args.val_video
    
    create_train_data(target_data_sub_dir, dest_sub_dir, exp_name, train_video,  test_video, val_video, feature_method)
    create_test_data(target_data_sub_dir, dest_sub_dir, exp_name, test_video_list, feature_method)
    