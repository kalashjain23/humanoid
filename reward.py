def forward_velocity(data):
    return data.qvel[0] # linear acceleration in x-axis

def upright(data):
    return data.body("Waist").xmat[8] # 1.0 when it is upright, -1.0 when upside down