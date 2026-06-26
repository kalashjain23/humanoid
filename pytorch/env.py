import numpy as np
import mujoco
from reward import (
    upright,
    height_reward,
    control_cost,
)


CONTROLLABLE_JOINTS = [
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


def randomize_state(model, data):
    """Randomize trunk pose and controllable joints, leave arms/head at keyframe."""
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # randomize trunk orientation: small random tilt around random axis
    angle = np.random.uniform(-0.15, 0.15)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    data.qpos[3] = np.cos(angle / 2.0)
    data.qpos[4] = axis[0] * np.sin(angle / 2.0)
    data.qpos[5] = axis[1] * np.sin(angle / 2.0)
    data.qpos[6] = axis[2] * np.sin(angle / 2.0)

    # randomize trunk height slightly
    data.qpos[2] += np.random.uniform(-0.05, 0.05)

    # randomize only the 13 controllable joints (waist + legs)
    for name in CONTROLLABLE_JOINTS:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr] += np.random.uniform(-0.15, 0.15)

    mujoco.mj_forward(model, data)


def get_foot_contacts(model, data):
    """
    Return (left_contact, right_contact) as booleans.
    Checks every active contact and asks whether either of its two geoms
    belongs to a foot body.
    """
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
    """
    Per-frame observation = 10 (IMU) + 13 (qpos) + 13 (qvel) + 3 (projected gravity) = 39.
    Stacked 3 frames in the caller -> 117 dims of base obs.
    Caller appends prev_action (13) and command (3) -> 133 total.
    """
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


def _foot_world_xy_speed(model, data, body_name):
    """World-frame horizontal speed of a body (for the slip cost)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
    return float(np.linalg.norm(vel[3:5]))


def walk_reward_function(model, data, command,
                         air_times, first_contact, current_contact, phase, cfg):
    """
    MuJoCo Playground T1 joystick reward, ported to numpy/MuJoCo.

    Positive: velocity tracking (sigma 0.25), gait-phase tracking, alive bonus,
    feet air time. Everything else is a small cost (orientation, base rocking,
    foot slip, foot distance, pose deviation, joint limits, self collision).

    phase: np.array([phase_L, phase_R]) in [-pi, pi] - the gait clock.
    cfg: precomputed constants dict (default_pose, pose_weights, jnt_lower/upper,
         knee_idx, foot_z0, swing_height).
    """
    # Trunk rotation matrix (3x3). R.T rotates world -> base frame.
    R = data.body("Trunk").xmat.reshape(3, 3)

    # --- velocity tracking in base frame, sigma 0.25 for both ---
    v_base = R.T @ data.qvel[:3]
    vx_base, vy_base = v_base[0], v_base[1]
    omega_base = R.T @ data.qvel[3:6]
    wz_base = omega_base[2]
    lin_sigma = 0.15   # tighter: kills lateral drift / sway
    ang_sigma = 0.25   # forgiving: a tight sigma flattens the yaw gradient when far off
    lin_err = (command[0] - vx_base) ** 2 + (command[1] - vy_base) ** 2
    ang_err = (command[2] - wz_base) ** 2
    tracking_lin_vel = float(np.exp(-lin_err / lin_sigma))
    tracking_ang_vel = float(np.exp(-ang_err / ang_sigma))

    # --- orientation + base rocking costs ---
    proj_grav = -R[2, :]                       # gravity in trunk frame
    orientation_cost = float(proj_grav[0] ** 2 + proj_grav[1] ** 2)
    ang_vel_xy_cost = float(omega_base[0] ** 2 + omega_base[1] ** 2)

    # --- feet air time: signed (penalizes short steps), clipped above at 0.3 ---
    cmd_norm = float(np.linalg.norm(command))
    if cmd_norm > 0.1:
        feet_air_time = float(np.sum(np.minimum(air_times - 0.2, 0.3) * first_contact.astype(float)))
    else:
        feet_air_time = 0.0

    # --- feet phase: track the gait clock's target foot height ---
    foot_z = np.array([
        data.body("left_foot_link").xpos[2],
        data.body("right_foot_link").xpos[2],
    ]) - cfg['foot_z0']
    rz = get_rz(phase, cfg['swing_height'])
    feet_phase = float(np.exp(-np.sum((foot_z - rz) ** 2) / 0.01))

    # --- feet distance: lateral separation in base yaw frame (fixes crossover) ---
    lf = data.body("left_foot_link").xpos
    rf = data.body("right_foot_link").xpos
    base_yaw = np.arctan2(R[1, 0], R[0, 0])
    feet_distance = abs(
        np.cos(base_yaw) * (lf[1] - rf[1]) - np.sin(base_yaw) * (lf[0] - rf[0])
    )
    feet_distance_cost = float(np.clip(0.2 - feet_distance, 0.0, 0.1))

    # --- feet slip: horizontal foot speed while in contact ---
    slip = 0.0
    if current_contact[0]:
        slip += _foot_world_xy_speed(model, data, "left_foot_link")
    if current_contact[1]:
        slip += _foot_world_xy_speed(model, data, "right_foot_link")
    feet_slip_cost = float(slip)

    # --- pose cost: weighted deviation from home (walking joints have zero weight) ---
    q = data.qpos[17:30]
    dq = q - cfg['default_pose']
    pose_cost = float(np.sum(cfg['pose_weights'] * dq ** 2))
    joint_dev_knee = float(np.sum(np.abs(dq[cfg['knee_idx']])))

    # --- soft joint position limits (95% of range) ---
    lower, upper = cfg['jnt_lower'], cfg['jnt_upper']
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * 0.95
    soft_lo, soft_hi = mid - half, mid + half
    dof_pos_limit_cost = float(np.sum(np.clip(soft_lo - q, 0.0, None) + np.clip(q - soft_hi, 0.0, None)))

    # --- self collision (feet touching each other) ---
    collision = 1.0 if get_foot_self_collision(model, data) else 0.0

    components = {
        "track_lin": 1.0 * tracking_lin_vel,
        "track_ang": 1.0 * tracking_ang_vel,
        "orient": -1.0 * orientation_cost,
        "ang_vel_xy": -0.15 * ang_vel_xy_cost,
        "air_time": 2.0 * feet_air_time,
        "feet_phase": 1.0 * feet_phase,
        "feet_dist": -1.0 * feet_distance_cost,
        "feet_slip": -0.25 * feet_slip_cost,
        "pose": -1.0 * pose_cost,
        "jdev_knee": -0.1 * joint_dev_knee,
        "dof_lim": -1.0 * dof_pos_limit_cost,
        "collision": -1.0 * collision,
        "alive": 0.25,
    }
    total = sum(components.values())
    return total, components

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