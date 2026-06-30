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


# Proportional high-level controller: world-frame goal -> base-frame [vx, vy, wz].

K_YAW = 2.5            # heading->wz gain, high so the 180 saturates to max turn
K_POS = 0.25           # distance->vx gain, slow forward to keep the turn arc tight
WZ_MAX = 0.5           # rad/s, matches training range
VX_MAX = 0.2           # m/s, creep forward so weak yaw can still come around
HEADING_GATE = 0.8     # rad, turn in place until roughly facing the goal
GOAL_RADIUS = 0.4      # m, declared reached early given mediocre tracking


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def high_level_command(data, goal_xy):
    """Returns (vx, vy, wz, distance)."""
    trunk_xy = data.body("Trunk").xpos[:2]
    R = data.body("Trunk").xmat.reshape(3, 3)
    current_heading = np.arctan2(R[1, 0], R[0, 0])

    delta = goal_xy - trunk_xy
    distance = float(np.linalg.norm(delta))
    desired_heading = float(np.arctan2(delta[1], delta[0]))
    heading_err = wrap_to_pi(desired_heading - current_heading)

    wz = float(np.clip(K_YAW * heading_err, -WZ_MAX, WZ_MAX))
    if abs(heading_err) > HEADING_GATE:
        vx = 0.0
    else:
        vx = float(np.clip(K_POS * distance, 0.0, VX_MAX))
    vy = 0.0
    return vx, vy, wz, distance


model = mujoco.MjModel.from_xml_path(str(ROOT / "booster_t1/scene.xml"))
data = mujoco.MjData(model)

actor = Actor()
checkpoint = torch.load(ROOT / "checkpoints/checkpoint_mjx_ft5.pt", map_location=torch.device('cpu'), weights_only=False)
actor.load_state_dict(checkpoint['actor'])
actor.eval()

obs_normalizer = RunningNormalizer(130)
obs_normalizer.mean = checkpoint['obs_mean']
obs_normalizer.var = checkpoint['obs_var']
obs_normalizer.count = checkpoint['obs_count']

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

dt = model.opt.timestep * 5

# gait clock (same as training): feet antiphase, fixed mid-range frequency
gait_freq = 1.5
phase = np.array([0.0, np.pi])
phase_dt = 2 * np.pi * dt * gait_freq

waypoints = [
    np.array([2.0, 0.0]),
]
wp_idx = 0
STOP_DECAY = 0.9       # per-step command decay once the final goal is reached
stopping = False
command = np.zeros(3)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # advance through waypoints; on the final one, ramp the command down to
        # zero instead of cutting it (the gait clock finishes its stride before
        # the freeze kicks in, so it settles into a stand rather than toppling)
        if not stopping:
            vx, vy, wz, dist = high_level_command(data, waypoints[wp_idx])
            if dist < GOAL_RADIUS:
                print(f"reached waypoint {wp_idx}")
                if wp_idx < len(waypoints) - 1:
                    wp_idx += 1
                    vx, vy, wz, dist = high_level_command(data, waypoints[wp_idx])
                else:
                    stopping = True
            command = np.array([vx, vy, wz])
        else:
            command = command * STOP_DECAY

        raw_obs = np.concatenate(obs_history)
        to_normalize = np.concatenate([raw_obs, prev_action])
        norm_part = obs_normalizer.normalize(to_normalize)
        phase_obs = np.concatenate([np.cos(phase), np.sin(phase)])
        full_obs = np.concatenate([norm_part, command, phase_obs])   # 137
        stacked_obs = torch.tensor(full_obs, dtype=torch.float32)

        with torch.no_grad():
            raw_mean = actor(stacked_obs)
        smoothed_mean = alpha * raw_mean + (1 - alpha) * prev_mean
        prev_mean = smoothed_mean

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

        # sync to real time
        elapsed = time.time() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
