import numpy as np
import mujoco

# velocity is tracked as windowed base displacement, not instantaneous qvel, so trunk
# sway can't fake forward motion. VEL_WINDOW = steps of base-xy history.
VEL_WINDOW = 50

# reference-state init: half the episodes start in single support, one foot lifted to its
# swing apex, seeding the states a standing reset never reaches.
SS_PROB = 0.5
SWING_LIFT = np.array([-0.4, 0.9, 0.25])   # hip_pitch, knee, ankle_pitch deltas
LEFT_LEG = np.array([1, 4, 5])             # indices into qpos[17:30]
RIGHT_LEG = np.array([7, 10, 11])


def randomize_state(model, data):
    """Randomize trunk pose and controllable joints, leave arms/head at keyframe.
    For SS_PROB of episodes, lift one leg into its swing apex (reference-state init).
    Returns (single, swing_left) so the caller can set the matching gait phase."""
    mujoco.mj_resetDataKeyframe(model, data, 0)

    angle = np.random.uniform(-0.15, 0.15)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    data.qpos[3] = np.cos(angle / 2.0)
    data.qpos[4] = axis[0] * np.sin(angle / 2.0)
    data.qpos[5] = axis[1] * np.sin(angle / 2.0)
    data.qpos[6] = axis[2] * np.sin(angle / 2.0)

    data.qpos[2] += np.random.uniform(-0.05, 0.05)
    data.qpos[17:30] += np.random.uniform(-0.15, 0.15, size=13)

    single = np.random.random() < SS_PROB
    swing_left = np.random.random() < 0.5
    if single:
        legs = LEFT_LEG if swing_left else RIGHT_LEG
        data.qpos[17 + legs] += SWING_LIFT

    mujoco.mj_forward(model, data)
    return single, swing_left


def base_frame_velocity(data, pos_hist, dt, window=VEL_WINDOW):
    """Windowed displacement velocity rotated into the base (trunk-yaw) frame.
    pos_hist is the (window, 2) history of base xy; pos_hist[0] is the oldest sample."""
    base_xy = data.qpos[:2]
    R = data.body("Trunk").xmat.reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    vel_world = (base_xy - pos_hist[0]) / (window * dt)
    return np.array([
        np.cos(yaw) * vel_world[0] + np.sin(yaw) * vel_world[1],
        -np.sin(yaw) * vel_world[0] + np.cos(yaw) * vel_world[1],
    ])


def initial_phase(single, swing_left):
    """Gait-clock phase at reset: lifted foot at swing apex (phase 0) for single-support
    starts, a random antiphase point otherwise. Mirrors mjx/mjx_env.py:reset."""
    if single:
        return np.array([0.0, np.pi]) if swing_left else np.array([np.pi, 0.0])
    theta = np.random.uniform(0.0, 2 * np.pi)
    return np.fmod(np.array([theta, theta + np.pi]) + np.pi, 2 * np.pi) - np.pi


def get_foot_contacts(model, data):
    """Return (left_contact, right_contact): true if any active contact involves that foot body."""
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")

    left_contact = False
    right_contact = False
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 == left_id or b2 == left_id:
            left_contact = True
        if b1 == right_id or b2 == right_id:
            right_contact = True
    return left_contact, right_contact

def get_observation(model, data):
    """Per-frame obs = 10 IMU + 13 qpos + 13 qvel + 3 projected gravity = 39.
    Caller stacks 3 frames then appends prev_action and command."""
    sensors_data = np.array([])
    for i in range(model.nsensor):
        start = model.sensor_adr[i]
        end = model.sensor_adr[i] + model.sensor_dim[i]
        sensors_data = np.append(sensors_data, data.sensordata[start:end])

    # Projected gravity: world gravity [0,0,-1] expressed in trunk frame.
    # R.T @ [0,0,-1] = -R[2, :], i.e. the negated third row of the trunk rotation matrix.
    proj_gravity = -data.body("Trunk").xmat[6:9]

    obs = np.concatenate([
        sensors_data,
        data.qpos[17:30],
        data.qvel[16:29],
        proj_gravity,
    ])
    return obs


def get_rz(phi, swing_height=0.08):
    """Target foot height as a function of gait phase (cubic bezier swing profile).
    phi in [-pi, pi]: 0 at stance, peaks at swing_height mid-swing. Vectorized over feet."""
    def bezier(y_start, y_end, x):
        return y_start + (y_end - y_start) * (x ** 3 + 3 * (x ** 2 * (1 - x)))
    x = (phi + np.pi) / (2 * np.pi)
    stance = bezier(0.0, swing_height, 2 * x)
    swing = bezier(swing_height, 0.0, 2 * x - 1)
    return np.where(x <= 0.5, stance, swing)


def get_foot_self_collision(model, data):
    """True if a contact exists directly between the left and right foot bodies."""
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if (b1 == left_id and b2 == right_id) or (b1 == right_id and b2 == left_id):
            return True
    return False


def walk_reward_function(model, data, command, lin_vel, action, prev_action,
                         air_times, first_contact, current_contact, phase,
                         prev_foot_xy, cfg):
    """Walk reward (numpy mirror of mjx/mjx_env.py:reward).
    lin_vel is windowed base-frame velocity; phase is the gait clock [phase_L, phase_R]."""
    R = data.body("Trunk").xmat.reshape(3, 3)

    # gate tracking on uprightness so the policy can't bank reward while toppling
    omega_base = R.T @ data.qvel[3:6]
    lin_err = (command[0] - lin_vel[0]) ** 2 + (command[1] - lin_vel[1]) ** 2
    ang_err = (command[2] - omega_base[2]) ** 2
    track_lin = float(np.exp(-lin_err / 0.25))
    track_ang = float(np.exp(-ang_err / 0.1))   # tight kernel: demand precise yaw match
    upright = float(np.clip(R[2, 2], 0.0, 1.0))
    track_lin *= upright
    track_ang *= upright

    orient = float(R[2, 0] ** 2 + R[2, 1] ** 2)
    ang_vel_xy = float(omega_base[0] ** 2 + omega_base[1] ** 2)

    # air-time bonus per landing once past a short threshold, only while moving
    cmd_norm = float(np.linalg.norm(command))
    air = float(np.sum(np.clip(air_times - 0.02, 0.0, 0.3) * first_contact.astype(float)))
    feet_air = air if cmd_norm > 0.1 else 0.0

    foot_z = np.array([
        data.body("left_foot_link").xpos[2],
        data.body("right_foot_link").xpos[2],
    ]) - cfg['foot_z0']
    feet_phase = float(np.exp(-np.sum((foot_z - get_rz(phase, cfg['swing_height'])) ** 2) / 0.01))

    # feet slip: horizontal foot displacement while planted (kills the skating cheat)
    foot_xy = np.array([
        data.body("left_foot_link").xpos[:2],
        data.body("right_foot_link").xpos[:2],
    ])
    foot_speed = np.linalg.norm((foot_xy - prev_foot_xy) / cfg['dt'], axis=1)
    feet_slip = float(np.sum(foot_speed * current_contact.astype(float)))

    action_rate = float(np.sum((action - prev_action) ** 2))
    q = data.qpos[17:30]
    pose_cost = float(np.sum(cfg['pose_weights'] * (q - cfg['default_pose']) ** 2))
    lower, upper = cfg['jnt_lower'], cfg['jnt_upper']
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * 0.95
    dof_lim = float(np.sum(np.clip(mid - half - q, 0.0, None) + np.clip(q - mid - half, 0.0, None)))

    collision = 1.0 if get_foot_self_collision(model, data) else 0.0

    components = {
        "track_lin": 2.0 * track_lin,
        "track_ang": 2.5 * track_ang,
        "feet_air": 4.0 * feet_air,
        "feet_phase": 1.0 * feet_phase,
        "alive": 0.25,
        "orient": -1.0 * orient,
        "ang_vel_xy": -0.15 * ang_vel_xy,
        "feet_slip": -1.0 * feet_slip,
        "action_rate": -0.05 * action_rate,
        "pose": -1.0 * pose_cost,
        "dof_lim": -1.0 * dof_lim,
        "collision": -1.0 * collision,
    }
    total = sum(components.values())
    return total, components

def check_termination(data):
    """End the rollout if the trunk falls too low, tips over, or leans sideways too far."""
    trunk_z = data.body("Trunk").xpos[2]
    trunk_upright = data.body("Trunk").xmat[8]
    trunk_lean = abs(data.body("Trunk").xmat[6])
    return trunk_z < 0.55 or trunk_upright < 0.7 or trunk_lean > 0.3