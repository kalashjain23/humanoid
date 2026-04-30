import numpy as np
import math


def upright(data):
    trunk_up = data.body("Trunk").xmat[8]
    return max(0, min(1, (trunk_up - 0.8) / (0.98 - 0.8))) # 0 at 0.8, 1 at 0.98

def height_reward(data):
    trunk_z = data.body("Trunk").xpos[2]
    return max(0, min(1, (trunk_z - 0.5) / (0.65 - 0.5))) # 0 at 0.5, 1 at 0.65

def control_cost(data):
    ctrl = float(np.sum(np.square(data.ctrl)))
    return max(0, 1.0 - 0.001 * ctrl)
    
def balance_reward(data):
    com_x = data.subtree_com[1][0]
    com_y = data.subtree_com[1][1]
    
    left_foot_x = data.body("left_foot_link").xpos[0]
    left_foot_y = data.body("left_foot_link").xpos[1]
    right_foot_x = data.body("right_foot_link").xpos[0]
    right_foot_y = data.body("right_foot_link").xpos[1]
    
    mid_x = (left_foot_x + right_foot_x) / 2.0
    mid_y = (left_foot_y + right_foot_y) / 2.0
    
    dist = math.sqrt((com_x - mid_x)**2 + (com_y - mid_y)**2)
    
    return max(0, 1.0 - dist / 0.15)