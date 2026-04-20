import numpy as np


def forward_velocity(data):
    return data.qvel[0] # linear acceleration in x-axis

def upright(data):
    trunk_up = data.body("Trunk").xmat[8]
    return max(0, min(1, (trunk_up - 0.8) / (0.98 - 0.8))) # 0 at 0.8, 1 at 0.98

def height_reward(data):
    trunk_z = data.body("Trunk").xpos[2]
    return max(0, min(1, (trunk_z - 0.5) / (0.65 - 0.5))) # 0 at 0.5, 1 at 0.65

def control_cost(data):
    ctrl = float(np.sum(np.square(data.ctrl)))
    return max(0, 1.0 - 0.001 * ctrl)