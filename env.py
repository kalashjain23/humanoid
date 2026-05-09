import numpy as np
from reward import (
    upright,
    height_reward,
    control_cost,
)


def get_observation(model, data):
    """
    Preparing observations that contains the IMU, waist and legs data
    to feed the networks.
    
    Total obs per frame = 10 (IMU) + 13 (qpos) + 13 (qvel) = 36 dims.
    We stack 3 frames so the network sees motion history = 108 dims.
    
    qpos[17:30] and qvel[16:29] pick out the waist + leg joints only.
    qpos layout: 7 free-joint dims (trunk pose), then 10 head/arm joints,
    then waist + legs start at index 17.
    qvel layout: 6 free-joint dims (trunk velocity), then head/arm velocities,
    then waist + leg velocities start at index 16.
    """
    sensors_data = np.array([])
    for i in range(model.nsensor):
        start = model.sensor_adr[i]
        end = model.sensor_adr[i]+model.sensor_dim[i]
        sensors_data = np.append(sensors_data, data.sensordata[start:end])
        
    obs = np.concatenate([sensors_data, data.qpos[17:30], data.qvel[16:29]])
    
    return obs

def stand_reward_function(data, step):
    """
    Reward function to train the humanoid to stand.
    - rewards being upright, keeping the waist above a particular height.
    - penalizes strong control commands.
    """
    stand_reward = height_reward(data) * upright(data) * control_cost(data)
    survival_bonus = min(step / 1000.0, 1.0)  # increases from 0 to 1 over 1000 steps to promote being alive

    return (
        5.0 * stand_reward
    ) * (1.0 + survival_bonus)

def check_termination(data):
    """
    Termination conditions to end the rollout
    - trunk_z: height of the trunk
    - trunk_upright: how upright the trunk is (1.0 = perfectly vertical)
    - trunk_lean: sideways lean of the trunk
    """
    trunk_z = data.body("Trunk").xpos[2]
    trunk_upright = data.body("Trunk").xmat[8]
    trunk_lean = abs(data.body("Trunk").xmat[6])
    return trunk_z < 0.45 or trunk_upright < 0.7 or trunk_lean > 0.25