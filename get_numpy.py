import numpy as np
import math
import glob
import utils
import argparse
import cv2

from parse import parse_sequence, load_ps
from pose_estimation import get_keypoints
from pose_trainer import evaluate_npy
from parse import parse_sequence, load_ps
from evaluate import evaluate_pose
from pprint import pprint

# run estimation & pose-trainer
# 1. video split to image
# 2. use estimation function to get numpy file
# 3. numpy file put pose-trainer evalutation fuction

# 후에 pose-trainer main을 바로 동영상을 넣으면 실행할 수 있도록 바꾸기
# video file capture

test_frame = 20

def capture_video(video): # .mp4 to .jpg
    cap = cv2.VideoCapture(video)
    count = 0

    save_path = "./cap_imgs"

    while(cap.isOpened()):
        ret, image = cap.read()

        # 캡쳐된 이미지를 저장하는 함수 
        cv2.imwrite(f"{save_path}/frame_{count}.jpg", image)
        
        print('Saved frame%d.jpg' % count)
        count += 1

        # end while loop after saved test_frame frame
        if count == test_frame : break
        
    cap.release()

def get_numpy(file_path, save_path): # file path 안에 있는 모든 .mp4 -> .npy 로 변환 
    files = sorted(glob.glob(file_path + '*.mp4'))
    for fname in files:
        pass
    print(files)

    for file in files:
        print(file)
        capture_video(file)
        print("\n\nSTART\n\n")
        keypoints = []

        for count in range(test_frame) :
            print('frame :', count)
            keypoints.append(get_keypoints(f'./cap_imgs/frame_{count}.jpg'))
#             print(keypoints)

        keypoints = np.array(keypoints)
        print('./numpy/' + save_path + '/' + file[len(file_path):-4])
        np.save('./numpy/' + save_path + '/' + file[len(file_path):-4], keypoints)   
