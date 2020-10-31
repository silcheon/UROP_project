# run estimation & pose-trainer
# 1. video split to image
# 2. use estimation function to get numpy file
# 3. numpy file put pose-trainer evalutation fuction

# 후에 pose-trainer main을 바로 동영상을 넣으면 실행할 수 있도록 바꾸기
# video file capture

import cv2
import numpy as np
#from pose_estimation.pose_estimation import get_keypoints
#from pose_trainer.pose_trainer import evaluate_npy
from pose_estimation import get_keypoints
from pose_trainer import evaluate_npy

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



def check_differentitaion_angle() :
    
    pass
    
if __name__ == '__main__' :
    # save capture files for testing
    # video_path = './video/bicep_curl.mp4'
    # capture_video(video_path)

    print("\n\nSTART\n\n")
    keypoints = []

    #get_keypoints('./images/p1.jpg')
    
    for count in range(test_frame) :
        print('frame :', count)
        keypoints.append(get_keypoints(f'./cap_imgs/frame_{count}.jpg'))
        
        
    keypoints = np.array(keypoints)
    np.save('./numpy/test', keypoints)
    
    evaluate_npy('./numpy/test.npy', 'bicep_curl')


