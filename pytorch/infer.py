import mujoco
import mujoco.viewer
import torch
import numpy as np
import time
from pathlib import Path

from model import Actor
from running_normalizer import RunningNormalizer
from env import get_observation

ROOT = Path(__file__).resolve().parents[1]

model = mujoco.MjModel.from_xml_path(str(ROOT / "booster_t1/scene.xml"))
data = mujoco.MjData(model)

actor = Actor()
checkpoint = torch.load(ROOT / "checkpoints/checkpoint_mjx_ft5.pt", map_location=torch.device('cpu'), weights_only=False)
actor.load_state_dict(checkpoint['actor'])
actor.eval()

# 130-dim: 117 stacked base + 13 prev_action
obs_normalizer = RunningNormalizer(130)
obs_normalizer.mean = checkpoint['obs_mean']
obs_normalizer.var = checkpoint['obs_var']
obs_normalizer.count = checkpoint['obs_count']

# base-frame command [vx, vy, wz]: [0,0,0] stand, [0.3,0,0] forward, [0,0,0.5] turn
command = np.array([0.5, 0.0, 0.0])

ctrl_low = torch.tensor(model.actuator_ctrlrange[:, 0], dtype=torch.float32)
ctrl_high = torch.tensor(model.actuator_ctrlrange[:, 1], dtype=torch.float32)

mujoco.mj_resetDataKeyframe(model, data, 0)
home_ctrl = torch.tensor(data.ctrl.copy(), dtype=torch.float32)
mujoco.mj_forward(model, data)

first_obs = get_observation(model, data)
obs_history = [first_obs, first_obs, first_obs]
prev_action = np.zeros(13)

prev_mean = torch.zeros(13)
alpha = 1.0   # no smoothing: match training, EMA lag destabilized the gait

dt = model.opt.timestep * 5  # 5 frame skip = 0.01s per policy step

# gait clock (same as training): feet antiphase, fixed mid-range frequency
gait_freq = 1.5
phase = np.array([0.0, np.pi])
phase_dt = 2 * np.pi * dt * gait_freq

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 117 stacked + 13 prev_action -> normalize -> + 3 command + 4 phase = 137
        raw_obs = np.concatenate(obs_history)
        to_normalize = np.concatenate([raw_obs, prev_action])
        norm_part = obs_normalizer.normalize(to_normalize)
        phase_obs = np.concatenate([np.cos(phase), np.sin(phase)])
        full_obs = np.concatenate([norm_part, command, phase_obs])
        stacked_obs = torch.tensor(full_obs, dtype=torch.float32)

        with torch.no_grad():
            raw_mean = actor(stacked_obs)
        smoothed_mean = alpha * raw_mean + (1 - alpha) * prev_mean
        prev_mean = smoothed_mean

        # action is a delta on home control (same scaling as training)
        action_scaled = home_ctrl.clone()
        action_scaled[10:23] += smoothed_mean * 0.5
        action_scaled = torch.clamp(action_scaled, ctrl_low, ctrl_high)
        data.ctrl[:] = action_scaled.numpy()

        for _ in range(5):
            mujoco.mj_step(model, data)

        new_obs = get_observation(model, data)
        obs_history.pop(0)
        obs_history.append(new_obs)
        prev_action = smoothed_mean.numpy()

        # advance gait clock, freeze when standing
        phase = phase + phase_dt
        phase = np.fmod(phase + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(command) <= 0.01:
            phase = np.array([np.pi, np.pi])

        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)