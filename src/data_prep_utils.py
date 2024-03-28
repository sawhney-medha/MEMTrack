import zipfile
import openpyxl
import os
import csv
import sys
import cv2
import shutil
import PIL
import glob
import pandas as pd
import numpy as np
from natsort import natsorted
from PIL import Image
import argparse

def load_map(video_map_path):
    # Initialize an empty dictionary
    video_dict = {}

    # Open and read the text file
    with open(video_map_path, 'r') as file:
        next(file)
        for line in file:
            #print(line)
            # Split each line into ID and name using the comma as a separator
            video_id, video_name = line.strip().split(',')

            # Convert the ID to an integer if needed
            video_id = int(video_id)

            # Add the key-value pair to the dictionary
            video_dict[video_id] = video_name

    # Print the loaded dictionary
#     print(video_dict)
    return video_dict

#find video number from video map
def find_next_video_num(data_dir):
    #data_dir = "../Data/"
    data_sub_dirs = glob.glob(f'{data_dir}/*')  
    print("data dirs: ", data_sub_dirs)
    video_dirs = glob.glob(f'{data_dir}/*/*')  
    max_id = 0
    for video in video_dirs:
        #print(video)
        if "ipynb" not in video:
            video_id = int(video.split("video")[1])
            #print(video_id)
            max_id = max(video_id, max_id)
    print("last video num: ",max_id)
    return max_id +1

def video_in_videomap(video_map_path, folder):
   # load videomap
    video_list = [value.strip() for value in load_map(video_map_path).values()]
    video_list = [value.lstrip("_") for value in video_list]
    
    if os.path.basename(folder.split(".zip")[0]) in video_list:
        return True
        
def add_video_to_videomap(video_map_path, video_dir, final_data_dir):
    video_num = find_next_video_num(final_data_dir)
    print("curr video num: ",video_num)
    
    video_map = open(video_map_path,'a+')
    video_strain = os.path.basename(video_dir.strip().strip("/"))
    video_map.write("\n")
    video_map.write(f"{video_num}, {video_strain}")
    video_map.close()
    
    print("Added to videomap")
    return video_num
    
    
def create_annotations(video_dir, csv_file_name, inference_mode, video_num, target_data_sub_dir ):
    count = 0
    images_source = os.path.join(video_dir, "Images without Labels")
    files = natsorted(os.listdir(images_source))
#     files[:5]
    data_dir = video_dir
    all_files = os.listdir(data_dir)
    if not inference_mode:
        try:
            #csv_file_name = list(filter(lambda f: f.endswith('Tracked.csv'), all_files))[0] if list(filter(lambda f: f.endswith('.csv'), all_files)) else list(filter(lambda f: f.endswith('.xlsx'), all_files))[0]
            #csv_file_name = "Raw Data.csv"
            csv_file = os.path.join(data_dir, csv_file_name)
#             print(csv_file_name)
            if ".csv" not in csv_file_name:
                read_file = pd.read_excel(csv_file, engine='openpyxl')
                read_file.to_csv (csv_file.split("xlsx")[0]+".csv", 
                              index = None, header=True)
                csv_file = csv_file.split("xlsx")[0]+".csv"
#             print(csv_file)
            file = open(csv_file)
        except:
            print("No Raw Data present, assuming Inference mode only")
            inference_mode=True

    if not inference_mode:
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        # header = header[0].split("\t")
        print(header)
        header = [x.lower() for x in header]
        img_id = header.index('pid') #  header.index('t [Frame]') 
        x_id = header.index('x [pixel]')
        y_id = header.index('y [pixel]')
        try:
            sticking_id = header.index('sticking')
            subpop_id = header.index('subpopulation')
        except: 
            sticking_id = None
            subpop_id = None
        try:
            diff_id = header.index('e/h')

        except:
            diff_id = None
            print("No diff id found, assuming all easy")
#         print(img_id)
#         print(x_id)
#         print(y_id)
#         print(sticking_id)
#         print(subpop_id)

        rows = []
        for row in csvreader:
                rows.append(row)
        #rows
        file.close()
        
    dest_dir = os.path.join(target_data_sub_dir, f"video{video_num}/")
#     print("dest_dir: ", dest_dir)
    images_target =os.path.join(dest_dir, "frame1/images/")
    annotations_easy = os.path.join(dest_dir, "frame1/annotations_easy/")
    annotations_easy_hard = os.path.join(dest_dir, "frame1/annotations_easy_hard/")
    annotations_hard = os.path.join(dest_dir, "frame1/annotations_hard/")
    annotations_very_hard = os.path.join(dest_dir, "frame1/annotations_veryhard/")
    annotations_easy_hard_veryhard = os.path.join(dest_dir, "frame1/annotations_easy_hard_veryhard/")

    annotations_motility_high = os.path.join(dest_dir, "frame1/annotations_motility_high/")
    annotations_motility_low = os.path.join(dest_dir, "frame1/annotations_motility_low/")
    annotations_motility_wiggle = os.path.join(dest_dir, "frame1/annotations_motility_wiggle/")
    annotations_motility_mid = os.path.join(dest_dir, "frame1/annotations_motility_mid/")

    annotations_sticking_motile = os.path.join(dest_dir, "frame1/annotations_sticking_motile/")
    annotations_sticking_non_motile = os.path.join(dest_dir, "frame1/annotations_sticking_non_motile/")
    annotations_sticking_stick = os.path.join(dest_dir, "frame1/annotations_sticking_stick/")


    os.makedirs(images_target, exist_ok=True)
    os.makedirs(annotations_easy_hard, exist_ok=True)
    os.makedirs(annotations_easy_hard_veryhard, exist_ok=True)
    os.makedirs(annotations_hard, exist_ok=True)
    os.makedirs(annotations_easy, exist_ok=True)
    os.makedirs(annotations_very_hard, exist_ok=True)


    os.makedirs(annotations_sticking_stick, exist_ok=True)
    os.makedirs(annotations_sticking_motile, exist_ok=True)
    os.makedirs(annotations_sticking_non_motile, exist_ok=True)

    os.makedirs(annotations_motility_high, exist_ok=True)
    os.makedirs(annotations_motility_wiggle, exist_ok=True)
    os.makedirs(annotations_motility_mid, exist_ok=True)
    os.makedirs(annotations_motility_low, exist_ok=True)

    for i,image in enumerate(files):
        #copy and rename images
        image_name = str(count) +".tif"
        shutil.copy(os.path.join(images_source, image), os.path.join(images_target,image_name))
#         print(os.path.join(images_source, image))
#         print(os.path.join(images_target,image_name))
        #create annoatations txt file
        #image_id = int(files_track[i].split(")")[1].split(".tif")[0])
        txt_file = open(annotations_easy_hard + str(count) +".txt",'w')
        txt_file_easy = open(annotations_easy + str(count) +".txt",'w')
        txt_file_hard = open(annotations_hard + str(count) +".txt",'w')
        txt_file_very_hard = open(annotations_very_hard + str(count) +".txt",'w')
        txt_file_all = open(annotations_easy_hard_veryhard + str(count) +".txt",'w')

        txt_file_motility_high = open(annotations_motility_high + str(count) +".txt",'w')
        txt_file_motility_low = open(annotations_motility_low + str(count) +".txt",'w')
        txt_file_motility_wiggle = open(annotations_motility_wiggle  + str(count) +".txt",'w')
        txt_file_motility_mid = open(annotations_motility_mid + str(count) +".txt",'w')

        txt_file_sticking_stick = open(annotations_sticking_stick + str(count) +".txt",'w')
        txt_file_sticking_motile = open(annotations_sticking_motile + str(count) +".txt",'w')
        txt_file_sticking_non_motile = open(annotations_sticking_non_motile + str(count) +".txt",'w')

        if not inference_mode:
            for row in rows:
                # print(image_id)
                # print( row[img_id])
                ##print(row)
                if int(row[img_id])-1 == int(count):#PID starts from 1
                    # print(image_id)
                    # print( row[img_id])
                    txt_file_all.write(row[x_id])
                    txt_file_all.write(" ")
                    txt_file_all.write(row[y_id])
                    txt_file_all.write("\n")
                    try:
                        if row[diff_id]=="E":
                            txt_file_easy.write(row[x_id])
                            txt_file_easy.write(" ")
                            txt_file_easy.write(row[y_id])
                            txt_file_easy.write("\n")

                            txt_file.write(row[x_id])
                            txt_file.write(" ")
                            txt_file.write(row[y_id])
                            txt_file.write("\n")

                        if row[diff_id]=="H":
                            txt_file_hard.write(row[x_id])
                            txt_file_hard.write(" ")
                            txt_file_hard.write(row[y_id])
                            txt_file_hard.write("\n")

                            txt_file.write(row[x_id])
                            txt_file.write(" ")
                            txt_file.write(row[y_id])
                            txt_file.write("\n")                

                        if row[diff_id]=="VH":
                            txt_file_very_hard.write(row[x_id])
                            txt_file_very_hard.write(" ")
                            txt_file_very_hard.write(row[y_id])
                            txt_file_very_hard.write("\n")

                        if row[subpop_id]=="L":
                            txt_file_motility_low.write(row[x_id])
                            txt_file_motility_low.write(" ")
                            txt_file_motility_low.write(row[y_id])
                            txt_file_motility_low.write("\n")
                        if row[subpop_id]=="W":
                           #print("wiggle")
                            txt_file_motility_wiggle.write(row[x_id])
                            txt_file_motility_wiggle.write(" ")
                            txt_file_motility_wiggle.write(row[y_id])
                            txt_file_motility_wiggle.write("\n")
                        if row[subpop_id]=="M":
                            txt_file_motility_mid.write(row[x_id])
                            txt_file_motility_mid.write(" ")
                            txt_file_motility_mid.write(row[y_id])
                            txt_file_motility_mid.write("\n")
                        if row[subpop_id]=="H":
                            txt_file_motility_high.write(row[x_id])
                            txt_file_motility_high.write(" ")
                            txt_file_motility_high.write(row[y_id])
                            txt_file_motility_high.write("\n")

                        if row[sticking_id]=="S":
                            txt_file_sticking_stick.write(row[x_id])
                            txt_file_sticking_stick.write(" ")
                            txt_file_sticking_stick.write(row[y_id])
                            txt_file_sticking_stick.write("\n")
                        if row[sticking_id]=="M":
                            txt_file_sticking_motile.write(row[x_id])
                            txt_file_sticking_motile.write(" ")
                            txt_file_sticking_motile.write(row[y_id])
                            txt_file_sticking_motile.write("\n")
                        if row[sticking_id]=="NM":
                            txt_file_sticking_non_motile.write(row[x_id])
                            txt_file_sticking_non_motile.write(" ")
                            txt_file_sticking_non_motile.write(row[y_id])
                            txt_file_sticking_non_motile.write("\n")
                    except:
                        txt_file_easy.write(row[x_id])
                        txt_file_easy.write(" ")
                        txt_file_easy.write(row[y_id])
                        txt_file_easy.write("\n")

            txt_file.close()
        count = count+1
    print("Annotations processed")
    return inference_mode
    
    

#run only once to generate bacteria data
def generate_bacteria_data(file, video_dir):
    # Generate Bacteria Tracks specific data for Bacteria Analysis

    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    # header = header[0].split("\t")
    print(header)
    header = [x.lower() for x in header]
    img_id = header.index('pid') #  header.index('t [Frame]') 
    x_id = header.index('x [pixel]')
    y_id = header.index('y [pixel]')
    try:
        diff_id = header.index('e/h')
    except:
        diff_id = None
        print("No diff id found, assuming all easy")
    track_id = header.index('tid')
    print(img_id)
    print(x_id)
    print(y_id)
    print(track_id)

    rows = []
    for row in csvreader:
            rows.append(row)
    print(rows[:6])
    file.close()
    
    tid_visited = []
    #video_dir = "./video9_feature_optical_flow_median_back_2pyr_18win/test/"
    bacteria_folder = "bacteria"
    bacteria_easy_hard_state_file = "easy_hard_veryhard"
    bacteria_coords = "xy_coord"
    count = 0
    max_pid = 0
    prev_tid = 0

    for row in rows:
        pid = int(row[img_id])-1
        if max_pid < pid:
            max_pid = pid

    for row in rows:
        tid = row[track_id]
        if tid not in tid_visited:
            tid_visited.append(tid)

    #         if count<(max_pid-1) and count>0:
    #             # print(row)
    #             # print(rows[i+1])
    #             print(count)
    #             print(max_pid)
    #             for i in range((max_pid - count-1)):
    #                 txt_file.write(str(count))
    #                 txt_file.write(" ")
    #                 txt_file.write("NotPresent")
    #                 txt_file.write("\n")

    #                 coord_file = open(os.path.join(video_dir, bacteria_folder, str(prev_tid), bacteria_coords, str(count)) +".txt",'w')
    #                 coord_file.close()
    #                 count = count+1
            # txt_file.close()
            # coord_file.close()
            count = 0
            os.makedirs(os.path.join(video_dir, bacteria_folder, str(tid), bacteria_coords), exist_ok=True)
            #os.makedirs(os.path.join(video_dir, bacteria_folder, str(tid)), exist_ok=True)
            try:
                os.remove(os.path.join(video_dir, bacteria_folder, str(tid), bacteria_easy_hard_state_file) +".txt")
                #os.remove(os.path.join(video_dir, bacteria_folder, str(tid), bacteria_coords, str(count)) +".txt")
            except OSError:
                pass

        txt_file = open(os.path.join(video_dir, bacteria_folder, str(tid), bacteria_easy_hard_state_file) +".txt",'a')
        pid = int(row[img_id]) - 1
        if int(pid) == 0: #for optical flow since first frame is skipped
            continue
        if pid-2>count: # pid-1 because 1 is skipped
            # print(count)
            # print(pid)
            for i in range((pid - count-1)):
                txt_file.write(str(count))
                txt_file.write(" ")
                txt_file.write("NotPresent")
                txt_file.write("\n")

                coord_file = open(os.path.join(video_dir, bacteria_folder, str(tid), bacteria_coords, str(count)) +".txt",'w')
                coord_file.close()
                count = count+1

        txt_file.write(str(count))
        txt_file.write(" ")
        try:
            txt_file.write(row[diff_id])
        except:
            txt_file.write("E")
            #print("No diff id found, assuming all easy")
        txt_file.write("\n")



        coord_file = open(os.path.join(video_dir, bacteria_folder, str(tid), bacteria_coords, str(count)) +".txt",'a')
        coord_file.write(row[x_id])
        coord_file.write(" ")
        coord_file.write(row[y_id])
        coord_file.write("\n")


        count = count+1

    if count<(max_pid-1) and count>0:
            # print(row)
            # print(rows[i+1])
            print(count)
            print(max_pid)
            for i in range((max_pid - count)):
                txt_file.write(str(count))
                txt_file.write(" ")
                txt_file.write("NotPresent")
                txt_file.write("\n")

                coord_file = open(os.path.join(video_dir, bacteria_folder, str(tid), bacteria_coords, str(count)) +".txt",'w')
                coord_file.close()
                count = count+1
    txt_file.close()
    coord_file.close()

    
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


def process_data(folder, src, final_data_dir, out_sub_dir, videomap="videomap.txt", csv_file_name="Raw Data.csv", inference_mode=False, unzip=False):
   
    target_data_sub_dir = os.path.join(final_data_dir, out_sub_dir)
    print("target_data_sub_dir: ", target_data_sub_dir)
    os.makedirs(target_data_sub_dir, exist_ok=True)
    
    video_map_path = os.path.join(final_data_dir, videomap )
    print("video_map_path: ", video_map_path)
    
    video_dir = os.path.join(src, folder.split(".zip")[0])
    
    if unzip:
        # unzip data
        zip_data_dir = os.path.join(src, folder)
        with zipfile.ZipFile(zip_data_dir, 'r') as zip_ref:
            zip_ref.extractall(src)
    
    if os.path.exists(video_map_path) and video_in_videomap(video_map_path, folder):
        print("video alread processed: ", folder.split(".zip")[0])
        raise Exception("video alread processed")
        return
    
    # add video to videomap
    video_num = add_video_to_videomap(video_map_path, video_dir, final_data_dir)
    inference_mode = create_annotations(video_dir, csv_file_name, inference_mode, video_num, target_data_sub_dir)
   
    if not inference_mode:
        target_video_dir = f"{target_data_sub_dir}/video{video_num}/frame1"
        csv_file = os.path.join(video_dir, csv_file_name)
        file = open(csv_file)
        generate_bacteria_data(file, target_video_dir)
        file.close()
    
    data_path = target_data_sub_dir
    test_video = [f"video{video_num}"]
    for video in natsorted(test_video):
        if not video.startswith('.') and os.path.isdir(os.path.join(data_path,video))==True:
            for minivideo in natsorted(os.listdir(os.path.join(data_path,video))):
                if not minivideo.startswith('.') and os.path.isdir(os.path.join(data_path,video,minivideo)) == True:
                    #print(minivideo)
                    create_video(os.path.join(data_path,video,minivideo))
    
    print(f"Video {video_num} processed")
    return video_num
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process folders.")
    parser.add_argument("--src_dir", default = "DataAll/Collagen/", help="Path to the folder to process")
    parser.add_argument("--folder",  default ="video1", help="Path to the folder to process")
    parser.add_argument("--out_dir", default ="MEMTrack/data/",  help="Path to the folder to process")
    parser.add_argument("--out_sub_dir",   help="Path to the folder to process")
    parser.add_argument("--csv_file_name", help="Path to the folder to process", default= "Raw Data.csv")
    parser.add_argument("--videomap", default="videomap.txt")
    parser.add_argument("--no_gt", action="store_true")
    parser.add_argument("--unzip", action="store_true", help="Unzip folder")


    args = parser.parse_args(args)
    src = args.src_dir
    final_data_dir  = args.out_dir
    out_sub_dir = args.out_sub_dir
    videomap = args.videomap
    csv_file_name = args.csv_file_name
    inference_mode = args.no_gt
    folder= args.folder
    unzip = args.unzip    
    
    process_data(folder, src, final_data_dir, out_sub_dir, videomap, csv_file_name, inference_mode, unzip)