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


# ------------- high-level controller ----------------------------------------
# Converts (target_x, target_y) in world frame to a base-frame velocity command
# [vx, vy, wz] for the trained low-level policy. Proportional only - per SPEC.

K_YAW = 1.5            # heading -> wz gain (bumped to compensate weak track_ang)
K_POS = 0.4          # distance -> vx gain (slower forward so turns can keep up)
WZ_MAX = 0.5           # rad/s, matches training range
VX_MAX = 0.4           # m/s (smaller margin against fall)
HEADING_GATE = 0.6     # rad. Wider so we don't lock to turn-in-place too aggressively.
GOAL_RADIUS = 0.4      # m. Declare victory earlier given mediocre tracking.


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def high_level_command(data, goal_xy):
    """Pose -> velocity command. Returns (vx, vy, wz, distance)."""
    trunk_xy = data.body("Trunk").xpos[:2]
    R = data.body("Trunk").xmat.reshape(3, 3)
    # Robot forward axis (x-axis of trunk frame) in world.
    current_heading = np.arctan2(R[1, 0], R[0, 0])

    delta = goal_xy - trunk_xy
    distance = float(np.linalg.norm(delta))
    desired_heading = float(np.arctan2(delta[1], delta[0]))
    heading_err = wrap_to_pi(desired_heading - current_heading)

    wz = float(np.clip(K_YAW * heading_err, -WZ_MAX, WZ_MAX))
    if abs(heading_err) > HEADING_GATE:
        vx = 0.0   # turn in place until roughly facing the goal
    else:
        vx = float(np.clip(K_POS * distance, 0.0, VX_MAX))
    vy = 0.0
    return vx, vy, wz, distance


# ------------- setup --------------------------------------------------------
model = mujoco.MjModel.from_xml_path(str(ROOT / "booster_t1/scene.xml"))
data = mujoco.MjData(model)

actor = Actor()
checkpoint = torch.load(ROOT / "checkpoints/checkpoint_2500.pt", map_location=torch.device('cpu'), weights_only=False)
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

# EMA smoothing on policy mean to kill high-frequency chatter.
# alpha=1 -> no smoothing; lower = smoother + more lag. 0.3-0.7 reasonable.
prev_mean = torch.zeros(13)
alpha = 0.5

dt = model.opt.timestep * 5

# gait clock (same as training): feet antiphase, fixed mid-range frequency
gait_freq = 1.5
phase = np.array([0.0, np.pi])
phase_dt = 2 * np.pi * dt * gait_freq

# ------------- waypoints (xy in world frame) --------------------------------
waypoints = [
    np.array([3.0,  0.0]),
    np.array([3.0,  2.0]),
    np.array([0.0,  2.0]),
    np.array([0.0,  0.0]),
]
wp_idx = 0


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # ---- high-level: compute command from current pose + active goal --
        if wp_idx < len(waypoints):
            vx, vy, wz, dist = high_level_command(data, waypoints[wp_idx])
            if dist < GOAL_RADIUS:
                print(f"reached waypoint {wp_idx} at distance {dist:.2f}")
                wp_idx += 1
                command = np.array([0.0, 0.0, 0.0])
            else:
                command = np.array([vx, vy, wz])
        else:
            command = np.array([0.0, 0.0, 0.0])  # all done, stand

        # ---- low-level: trained policy ------------------------------------
        raw_obs = np.concatenate(obs_history)
        to_normalize = np.concatenate([raw_obs, prev_action])
        norm_part = obs_normalizer.normalize(to_normalize)
        phase_obs = np.concatenate([np.cos(phase), np.sin(phase)])
        full_obs = np.concatenate([norm_part, command, phase_obs])   # 137
        stacked_obs = torch.tensor(full_obs, dtype=torch.float32)

        with torch.no_grad():
            raw_mean = actor(stacked_obs)
        # one-step EMA on the mean. Magnitudes stay at training scale; only HF chatter is damped.
        smoothed_mean = alpha * raw_mean + (1 - alpha) * prev_mean
        prev_mean = smoothed_mean

        action_scaled = home_ctrl.clone()
        action_scaled[10:23] += smoothed_mean * 0.3
        action_scaled = torch.clamp(action_scaled, ctrl_low, ctrl_high)
        data.ctrl[:] = action_scaled.numpy()

        for _ in range(5):
            mujoco.mj_step(model, data)

        new_obs = get_observation(model, data)
        obs_history.pop(0)
        obs_history.append(new_obs)
        # feed the smoothed mean into prev_action so obs history stays consistent.
        prev_action = smoothed_mean.numpy()

        # advance the gait clock; freeze when commanded to stand
        phase = phase + phase_dt
        phase = np.fmod(phase + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(command) <= 0.01:
            phase = np.array([np.pi, np.pi])

        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
