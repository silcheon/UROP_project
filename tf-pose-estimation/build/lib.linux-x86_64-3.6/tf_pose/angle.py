import logging
import math

logger = logging.getLogger('AngleRun')

def angle_between_points(p0, p1, p2):
    a = (p1[0] - p0[0])**2 + (p1[1] - p0[1])**2
    b = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    c = (p2[0] - p0[0])**2 + (p2[1] - p0[1])**2
    
    return  math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi 

def length_between_points(p0, p1): 
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

def human_cnt(humans):
    if humans is None: 
        return 0
    return len(humans)

def get_angle_point(human, pos):
    pnts = []
    
    if pos == 'left_shoulder':
        pos_list = (1, 5, 6)
    elif pos == 'left_elbow':
        pos_list = (5, 6, 7)
    elif pos == 'left_pelvis': # 골반 
        pos_list = (1, 11, 12)
    elif pos == 'left_knee':
        pos_list = (11, 12, 13)
    elif pos == 'right_shoulder':
        pos_list = (1, 2, 3)
    elif pos == 'right_elbow':
        pos_list = (2, 3, 4)
    elif pos == 'right_pelvis': # 골반 
        pos_list = (1, 8, 9)
    elif pos == 'right_knee':
        pos_list = (8, 9, 10) 
    
    for i in range(3):
        if pos_list[i] not in human.body_parts.keys():
            logger.info('component [%d] incomplete', pos_list[i])
            return pnts
        p = human.body_parts[pos_list[i]]
        pnts.append((p.x, p.y))
    
    return pnts
 
def get_angle(human, pos):
    pnts = get_angle_point(human, pos)
    if len(pnts) != 3:
        logger.info('component incomplete')
        return 
    
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('%s angle: %f' % (pos, angle))
    return angle 

###    