# run estimation & pose-trainer
# 1. video split to image
# 2. use estimation function to get numpy file
# 3. numpy file put pose-trainer evalutation fuction

# 후에 pose-trainer main을 바로 동영상을 넣으면 실행할 수 있도록 바꾸기
# video file capture

import cv2
import numpy as np
import glob
#from pose_estimation.pose_estimation import get_keypoints
#from pose_trainer.pose_trainer import evaluate_npy
from pose_estimation import get_keypoints_for_cap
from pose_trainer import evaluate_npy

from tf_pose.networks import get_graph_path
from tf_pose.estimator import TfPoseEstimator


test_frame = 20

def capture_video(video) : 
    cap = cv2.VideoCapture('./video/bicep_curl.mp4')
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

#def make_numpy_file():
    

def check_differentitaion_angle() :    
    pass
    
if __name__ == '__main__' :
    
    # # save capture files for testing
    # video_path = './video/squat/squat_good_1.mp4'
    # capture_video(video_path)

    # print("\n\nSTART\n\n")
    # keypoints = []

    # #get_keypoints('./images/p1.jpg')
    
    # for count in range(test_frame) :
    #     print('frame :', count)
    #     keypoints.append(get_keypoints(f'./cap_imgs/frame_{count}.jpg'))
        
    # keypoints = np.array(keypoints)
    # np.save('./numpy/test', keypoints)
    
    # evaluate_npy('./numpy/test.npy', 'bicep_curl')


    ######################################################################
    # CONVERT VIDEO TO NUMPY
    ######################################################################
    # for capture
    exercise = 'squat'
    file_path = './video/squat/'
    save_path = "./poses_compressed/squat"

    files = sorted(glob.glob(file_path + '*.m*'))
    # 또는 변환할 파일들
    
    print(files)
    #files = ['squat_good_1.mp4']

    # 이미 변환된 파일 제외
    saved_files = sorted(glob.glob(save_path + '/*.npy'))
    saved_files = [f.split('/')[-1][:-4] for f in saved_files]
    print(saved_files)

    #files = files
    for f in files[:] :
        fn = f.split('/')[-1][:-4]
        if fn in saved_files : continue

        keypoints = []
        cap = cv2.VideoCapture(f)

        # need image, model, resize, resize-out-ratio
        w, h = 432, 368          # image size fixed 432x368
        model='mobilenet_thin'
        
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

        print(f"\n\nStart Converting {f}")
        fcnt = 0
        while (cap.isOpened()):
            fcnt+=1

            ret, image = cap.read()
            k = get_keypoints_for_cap(image, e, False)

            if k == -1 : break
            
            keypoints.append(k)
            print(f"converted {fcnt} frame")


        keypoints = np.array(keypoints)
        print(f"Converted {fcnt} frames")
        np.save(f'{save_path}/{fn}', keypoints)
        print(f"\n\nSaved {save_path}/{fn}")        

        

