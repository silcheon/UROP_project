import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import pprint # 
from tf_pose import angle # 각도 측정 모듈 


logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

### 옵션값 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args() 

    w, h = model_wh(args.resize) # 이미지 사이즈 값을 저장, w : 너비, h : 높이 
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368)) # graph의 check point를 불러옴. 
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
###

    ###
    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio) # inference, 입력된 이미지를 변환시킨 후 humans 에 저장  
    
    ###
#     print("\n\nhuman\n\n")
#     print(type(humans))
#     pprint.pprint(humans)
#     print(humans[0])
    ###
    # body part 값을 가지고 각도를 재서 모범이 되는 운동자세의 각도와 비교하면 될듯함. 비교할 body part는 팔, 몸통, 다리 부분으로 정하는 식으로 한다. 

    elapsed = time.time() - t  

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False) # draw_humans, 자세 추정 데이터(점, 선)를 그린 이미지를 저장  
    
    ### 각도 계산 추가 부분 
#     print(humans[0])
#     ang = angle.angle_left_elbow(humans[0])
    
    body_part_left_list = ['left_shoulder', 'left_elbow', 'left_pelvis', 'left_knee']
    
    body_part_right_list = ['right_shoulder', 'right_elbow', 'right_pelvis', 'right_knee']
    
    print("\n\n== body part left angle ==\n\n") 
    
    for i in body_part_left_list: 
        ang = angle.get_angle(humans[0], i)
        print('\n\n')
        print('%s angle is %f\n' % (i, ang))
        
        if ang > 130: print("Fold your arms a little!\n") # 예를 들어봄. 
    
    print('\n\n')
    
    print("\n\n== body part right angle ==\n\n") 
    
    for i in body_part_right_list: 
        ang = angle.get_angle(humans[0], i)
        print('\n\n')
        print('%s angle is %f\n' % (i, ang))
        
        if ang > 130: print("Fold your arms a little!\n") # 예를 들어봄. 
    
    print('\n\n')
    
    # TODO: 이미지에서 키포인트들의 좌표값과 coco 각 관절에 부여된 숫자 출력 
    ###
    
    # vectormap & heatmap plot 그리는 부분 
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Result')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show network output
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = e.pafMat.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        plt.show()
    except Exception as e:
        logger.warning('matplitlib error, %s' % e)
        cv2.imshow('result', image)
        cv2.waitKey()
    ###