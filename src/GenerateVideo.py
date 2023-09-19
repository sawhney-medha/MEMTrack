import cv2
import os
import argparse
from natsort import natsorted

ap = argparse.ArgumentParser(description='Training')
ap.add_argument('--video_num', default="", type=str, metavar='VIDEO')
ap.add_argument('--fps', default=1, type=int, metavar='FPS')
ap.add_argument('--data_path', default="19", type=str, metavar='PATH')
ap.add_argument('--custom_test_dir', type=str, metavar='CELL PATH')
args = ap.parse_args()
video_num = args.video_num
fps = args.fps

def create_video(data_dir, image_dir, video_name):
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    #print(data_dir)
    img_sample = cv2.imread(os.path.join(image_dir,"0.png"))
    #print(img_sample.shape)
    height, width, channels = img_sample.shape
    
    video = cv2.VideoWriter(data_dir + video_name + ".mp4", fourcc, fps, (width, height))
    #data_dir = "./Data/video3/"
    #image_dir = os.path.join(_dir, "images")
    for frame in natsorted(os.listdir(image_dir)):
        #print(frame)
        img = cv2.imread(os.path.join(image_dir, frame))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

if args.custom_test_dir:
    data_dir = args.custom_test_dir
else:
    data_dir = args.data_path + f"/data_video{video_num}_feature_optical_flow_median_back_2pyr_18win_background_img/"
image_dir = data_dir + "/test/tracklets-filtered/"
video_name = f'video{video_num}-tracklets-filtered-{fps}'
create_video(data_dir, image_dir, video_name)