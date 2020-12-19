import logging
import sys
import time

import cv2
import numpy as np
import pprint
import numpy as np
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

# get humans class
# model setted mobilenet_thin (later should check which model is the best and use)
def get_keypoints(image, model='mobilenet_thin', display=False) :
    cocopart_dict = {'Nose': 0,
                     'Neck': 1,  
                     'RShoulder' : 2,
                     'RElbow' : 3,
                     'RWrist' : 4,
                     'LShoulder' : 5,
                     'LElbow' : 6,
                     'LWrist' : 7,
                     'RHip' : 8,
                     'RKnee' : 9,
                     'RAnkle' : 10,
                     'LHip' : 11,
                     'LKnee' : 12,
                     'LAnkle' : 13,
                     'REye' : 14,
                     'LEye' : 15,
                     'REar' : 16,
                     'LEar' : 17
                                  } 

    key_points    = []
    
    # for logger
    '''
    logger = logging.getLogger('TfPoseEstimatorRun')
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    '''
    
    # need image, model, resize, resize-out-ratio
    w, h = 432, 368          # image size fixed 432x368
    upsample_size = 4.0      # default


    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))


    
    # if you use video capture, do not need line
    # estimate human poses from a single image !
    image = common.read_imgfile(image, None, None)

    if image is None :
        #logger.error('Image can not be read, path=%s' % args.image)
        print(f'Image {image} can not be read')
        sys.exit(-1)

    t = time.time()
    
    # upsample_size default setting 4.0
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=upsample_size)
    elapsed = time.time() - t
    
    #for key, value in cocopart_dict.items() :
    #    key_points[key] = (humans[0].body_parts[value].x, humans[0].body_parts[value].y)    
    for key, value in cocopart_dict.items() :
        # JS
        # 만약 추적된 Keypoints들이 없으면 key 값이 없다(제외시켜주기 위해 try 문 사용)
        try : 
            key_points.append([humans[0].body_parts[value].x, humans[0].body_parts[value].y, humans[0].body_parts[value].score])
        except : key_points.append([-1, -1, 0])

    #pprint.pprint(key_points)
        
    # display pose_estimation result
    if display == True :
        display_image(image, e)


    return key_points

def get_keypoints_for_cap(image, e, display=False) :
    cocopart_dict = {'Nose': 0,
                     'Neck': 1,  
                     'RShoulder' : 2,
                     'RElbow' : 3,
                     'RWrist' : 4,
                     'LShoulder' : 5,
                     'LElbow' : 6,
                     'LWrist' : 7,
                     'RHip' : 8,
                     'RKnee' : 9,
                     'RAnkle' : 10,
                     'LHip' : 11,
                     'LKnee' : 12,
                     'LAnkle' : 13,
                     'REye' : 14,
                     'LEye' : 15,
                     'REar' : 16,
                     'LEar' : 17
                                  } 

    key_points    = []

    w, h = 432, 368          # image size fixed 432x368
    upsample_size = 4.0      # default


    if image is None :
        #logger.error('Image can not be read, path=%s' % args.image)
        print(f'Image {image} can not be read')
        return -1

    t = time.time()
    
    # upsample_size default setting 4.0
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=upsample_size)
    elapsed = time.time() - t
    
    #for key, value in cocopart_dict.items() :
    #    key_points[key] = (humans[0].body_parts[value].x, humans[0].body_parts[value].y)    
    for key, value in cocopart_dict.items() :
        # JS
        # 만약 추적된 Keypoints들이 없으면 key 값이 없다(제외시켜주기 위해 try 문 사용)
        try : 
            key_points.append([humans[0].body_parts[value].x, humans[0].body_parts[value].y, humans[0].body_parts[value].score])
        except : key_points.append([-1, -1, 0])

    #pprint.pprint(key_points)
        
    # display pose_estimation result
    if display == True :
        display_image(image, e)


    return key_points


def display_image(image, e) :
        import matplotlib.pyplot as plt

        fig = plt.figure()

        bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show keypoints
        a = fig.add_subplot(2, 1, 1)
        a.set_title('keypoints')

        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.7)

        # show part affinity
        tmp2 = e.pafMat.transpose((2, 0, 1))
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 1, 2)
        a.set_title('affinity')

        plt.imshow(bgimg, alpha=0.5)
        #plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)

        plt.show()

        plt.waitforbuttonpress()
        plt.close()
        

if __name__ == '__main__':
    keypoints = {'LAnkle': (0.7083333333333334, 0.7771739130434783),
                 'LElbow': (0.6666666666666666, 0.3532608695652174),
                 'LHip': (0.6157407407407407, 0.46195652173913043),
                 'LKnee': (0.6712962962962963, 0.625),
                 'LShoulder': (0.5925925925925926, 0.21739130434782608),
                 'LWrist': (0.6620370370370371, 0.46195652173913043),
                 'Neck': (0.5046296296296297, 0.22282608695652173),
                 'RAnkle': (0.5277777777777778, 0.8260869565217391),
                 'RElbow': (0.36574074074074076, 0.2826086956521739),
                 'RHip': (0.49537037037037035, 0.4673913043478261),
                 'RKnee': (0.47685185185185186, 0.6032608695652174),
                 'RShoulder': (0.4212962962962963, 0.22282608695652173),
                 'RWrist': (0.27314814814814814, 0.32065217391304346)}
        
    keypoints = get_keypoints('./images/p1.jpg')
    #print((np.array([keypoints])).shape)
    np.save("./numpy/test", np.array([keypoints]))
    #print(f"x : {keypoints['RElbow'][0]}")
    #print(f"y : {keypoints['RElbow'][1]}")

    
 
