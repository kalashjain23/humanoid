import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from flax import struct
from pathlib import Path

def _scene_path():
    here = Path(__file__).resolve()
    for root in (here.parents[1], here.parent):   # local mjx/ layout, or flat on Modal
        p = root / "booster_t1/scene_mjx.xml"
        if p.exists():
            return str(p)
    raise FileNotFoundError("scene_mjx.xml not found")


N_SUBSTEPS = 5
SWING_HEIGHT = 0.08
RESAMPLE_EVERY = 250
VEL_WINDOW = 50        # steps of base-position history; track displacement velocity, not
                       # instantaneous qvel, so trunk sway can't fake forward motion

# reference-state init: half the episodes start in single support, one foot lifted to
# its swing apex, so the policy is placed in the states a standing reset never reaches.
SS_PROB = 0.5
SWING_LIFT = jp.array([-0.4, 0.9, 0.25])   # hip_pitch, knee, ankle_pitch deltas
LEFT_LEG = jp.array([1, 4, 5])             # indices into qpos[17:30]
RIGHT_LEG = jp.array([7, 10, 11])


@struct.dataclass
class Config:
    home_qpos: jp.ndarray
    home_ctrl: jp.ndarray
    default_pose: jp.ndarray
    foot_z0: jp.ndarray
    pose_weights: jp.ndarray
    jnt_lower: jp.ndarray
    jnt_upper: jp.ndarray
    ctrl_low: jp.ndarray
    ctrl_high: jp.ndarray
    left_geoms: jp.ndarray
    right_geoms: jp.ndarray
    trunk_id: int = struct.field(pytree_node=False)
    left_foot_id: int = struct.field(pytree_node=False)
    right_foot_id: int = struct.field(pytree_node=False)
    floor_id: int = struct.field(pytree_node=False)
    dt: float = struct.field(pytree_node=False)


@struct.dataclass
class State:
    data: mjx.Data
    obs_hist: jp.ndarray       # (3, 39)
    prev_action: jp.ndarray    # (13,)
    command: jp.ndarray        # (3,)
    phase: jp.ndarray          # (2,) gait clock
    phase_dt: jp.ndarray       # scalar
    air_time: jp.ndarray       # (2,)
    last_contact: jp.ndarray   # (2,) bool
    pos_hist: jp.ndarray       # (VEL_WINDOW, 2) base xy history
    prev_foot_xy: jp.ndarray   # (2, 2)
    rng: jp.ndarray
    step: jp.ndarray           # scalar int
    scale: jp.ndarray          # command curriculum scale


def load():
    """Build the mjx model and constant Config from the home keyframe."""
    mj_model = mujoco.MjModel.from_xml_path(_scene_path())
    mjx_model = mjx.put_model(mj_model)

    d = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, d, 0)
    mujoco.mj_forward(mj_model, d)

    bid = lambda n: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
    trunk, lf, rf = bid("Trunk"), bid("left_foot_link"), bid("right_foot_link")
    floor = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    geoms_of = lambda b: np.where(mj_model.geom_bodyid == b)[0]

    qadr = lambda nm: mj_model.jnt_qposadr[
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, nm)] - 17

    jnt_lower, jnt_upper = np.zeros(13), np.zeros(13)
    for j in range(mj_model.njnt):
        adr = mj_model.jnt_qposadr[j]
        if 17 <= adr < 30:
            jnt_lower[adr - 17] = mj_model.jnt_range[j, 0]
            jnt_upper[adr - 17] = mj_model.jnt_range[j, 1]

    pose_weights = np.ones(13)
    for nm in ["Left_Hip_Pitch", "Right_Hip_Pitch", "Left_Knee_Pitch",
               "Right_Knee_Pitch", "Left_Ankle_Pitch", "Right_Ankle_Pitch"]:
        pose_weights[qadr(nm)] = 0.0

    cfg = Config(
        home_qpos=jp.array(d.qpos.copy()),
        home_ctrl=jp.array(d.ctrl.copy()),
        default_pose=jp.array(d.qpos[17:30].copy()),
        foot_z0=jp.array([d.xpos[lf, 2], d.xpos[rf, 2]]),
        pose_weights=jp.array(pose_weights),
        jnt_lower=jp.array(jnt_lower),
        jnt_upper=jp.array(jnt_upper),
        ctrl_low=jp.array(mj_model.actuator_ctrlrange[:, 0]),
        ctrl_high=jp.array(mj_model.actuator_ctrlrange[:, 1]),
        left_geoms=jp.array(geoms_of(lf)),
        right_geoms=jp.array(geoms_of(rf)),
        trunk_id=trunk, left_foot_id=lf, right_foot_id=rf, floor_id=floor,
        dt=float(mj_model.opt.timestep * N_SUBSTEPS),
    )
    return mjx_model, cfg


def get_rz(phi, swing_height=SWING_HEIGHT):
    # target foot height over the gait phase: cubic bezier stance->swing->stance
    bez = lambda y0, y1, x: y0 + (y1 - y0) * (x ** 3 + 3 * (x ** 2 * (1 - x)))
    x = (phi + jp.pi) / (2 * jp.pi)
    stance = bez(0.0, swing_height, 2 * x)
    swing = bez(swing_height, 0.0, 2 * x - 1)
    return jp.where(x <= 0.5, stance, swing)


def _obs(data, cfg):
    proj_gravity = -data.xmat[cfg.trunk_id].reshape(3, 3)[2, :]
    return jp.concatenate([
        data.sensordata,          # gyro + accel + orientation quat
        data.qpos[17:30],
        data.qvel[16:29],
        proj_gravity,
    ])


def _contacts(data, cfg):
    """Per-foot ground contact as (2,) float, from the padded mjx contact buffer."""
    c = data._impl.contact
    touching = c.dist < 1e-3
    has_floor = jp.any(c.geom == cfg.floor_id, axis=1)
    has_left = jp.any(jp.isin(c.geom, cfg.left_geoms), axis=1)
    has_right = jp.any(jp.isin(c.geom, cfg.right_geoms), axis=1)
    left = jp.any(touching & has_floor & has_left)
    right = jp.any(touching & has_floor & has_right)
    self_col = jp.any(touching & has_left & has_right)
    return jp.array([left, right], dtype=jp.float32), self_col.astype(jp.float32)


def reward(data, command, lin_vel, action, prev_action, air_time, first_contact,
           current_contact, phase, prev_foot_xy, self_col, cfg):
    R = data.xmat[cfg.trunk_id].reshape(3, 3)

    # velocity tracking: lin_vel is the windowed base-frame displacement velocity, so
    # trunk sway can't fake net translation. yaw rate stays instantaneous.
    omega_base = R.T @ data.qvel[3:6]
    lin_err = (command[0] - lin_vel[0]) ** 2 + (command[1] - lin_vel[1]) ** 2
    ang_err = (command[2] - omega_base[2]) ** 2
    track_lin = jp.exp(-lin_err / 0.25)
    track_ang = jp.exp(-ang_err / 0.1)   # tighter kernel: demand precise yaw-rate match
    # gate tracking on uprightness so the policy can't bank velocity reward while
    # toppling - removes the lunge-and-fall incentive at the source.
    upright = jp.clip(R[2, 2], 0.0, 1.0)
    track_lin = track_lin * upright
    track_ang = track_ang * upright

    orient = R[2, 0] ** 2 + R[2, 1] ** 2
    ang_vel_xy = omega_base[0] ** 2 + omega_base[1] ** 2

    # gait reward: swing duration + bezier foot-height schedule from the clock
    cmd_norm = jp.linalg.norm(command)
    air = jp.sum(jp.clip(air_time - 0.02, 0.0, 0.3) * first_contact)
    feet_air = jp.where(cmd_norm > 0.1, air, 0.0)

    foot_z = jp.array([data.xpos[cfg.left_foot_id, 2],
                       data.xpos[cfg.right_foot_id, 2]]) - cfg.foot_z0
    feet_phase = jp.exp(-jp.sum((foot_z - get_rz(phase)) ** 2) / 0.01)

    # feet must not slide while planted (kills the skating cheat)
    foot_xy = jp.array([data.xpos[cfg.left_foot_id, :2], data.xpos[cfg.right_foot_id, :2]])
    foot_speed = jp.linalg.norm((foot_xy - prev_foot_xy) / cfg.dt, axis=1)
    feet_slip = jp.sum(foot_speed * current_contact)

    action_rate = jp.sum((action - prev_action) ** 2)
    q = data.qpos[17:30]
    pose_cost = jp.sum(cfg.pose_weights * (q - cfg.default_pose) ** 2)
    mid = 0.5 * (cfg.jnt_lower + cfg.jnt_upper)
    half = 0.5 * (cfg.jnt_upper - cfg.jnt_lower) * 0.95
    dof_lim = jp.sum(jp.clip(mid - half - q, 0.0, None) + jp.clip(q - mid - half, 0.0, None))

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
        "collision": -1.0 * self_col,
    }
    return sum(components.values()), components


def terminated(data, cfg):
    R = data.xmat[cfg.trunk_id].reshape(3, 3)
    trunk_z = data.xpos[cfg.trunk_id, 2]
    return (trunk_z < 0.55) | (R[2, 2] < 0.7) | (jp.abs(R[2, 0]) > 0.3)


def sample_command(rng, scale):
    # mix forward-walk with turn-in-place (vx=0) so the policy can rotate without
    # forward speed (goto.py commands vx=0 to face a goal); turning eased in by scale.
    k1, k2, k3 = jax.random.split(rng, 3)
    turn_in_place = jax.random.bernoulli(k3, 0.4)
    vx = jp.where(turn_in_place, 0.0,
                  jax.random.uniform(k1, (), minval=0.3, maxval=0.6))
    wz = jax.random.uniform(k2, (), minval=-0.5, maxval=0.5) * scale
    return jp.array([vx, 0.0, wz])


def reset(model, cfg, rng, scale):
    (k_tilt, k_ax, k_h, k_jnt, k_cmd, k_freq, k_theta,
     k_rsi, k_leg, rng) = jax.random.split(rng, 10)

    angle = jax.random.uniform(k_tilt, (), minval=-0.15, maxval=0.15)
    axis = jax.random.normal(k_ax, (3,))
    axis = axis / jp.linalg.norm(axis)
    quat = jp.concatenate([jp.cos(angle / 2)[None], axis * jp.sin(angle / 2)])

    qpos = cfg.home_qpos
    qpos = qpos.at[3:7].set(quat)
    qpos = qpos.at[2].add(jax.random.uniform(k_h, (), minval=-0.05, maxval=0.05))
    qpos = qpos.at[17:30].add(jax.random.uniform(k_jnt, (13,), minval=-0.15, maxval=0.15))

    # reference-state init: lift one leg into its swing apex for SS_PROB of episodes
    single = jax.random.bernoulli(k_rsi, SS_PROB)
    swing_left = jax.random.bernoulli(k_leg)
    lift = jp.where(swing_left, jp.zeros(13).at[LEFT_LEG].set(SWING_LIFT),
                    jp.zeros(13).at[RIGHT_LEG].set(SWING_LIFT))
    qpos = qpos.at[17:30].add(jp.where(single, lift, jp.zeros(13)))

    data = mjx.make_data(model).replace(qpos=qpos, ctrl=cfg.home_ctrl)
    data = mjx.forward(model, data)

    obs = _obs(data, cfg)
    base_xy = data.qpos[:2]
    foot_xy = jp.array([data.xpos[cfg.left_foot_id, :2], data.xpos[cfg.right_foot_id, :2]])
    freq = jax.random.uniform(k_freq, (), minval=1.25, maxval=1.75)
    # single-support starts: gait clock places the lifted foot at apex (phase 0), the
    # planted foot at pi. standing starts: a random point on the clock.
    theta = jax.random.uniform(k_theta, (), minval=0.0, maxval=2 * jp.pi)
    stand_phase = jp.fmod(jp.array([theta, theta + jp.pi]) + jp.pi, 2 * jp.pi) - jp.pi
    ss_phase = jp.where(swing_left, jp.array([0.0, jp.pi]), jp.array([jp.pi, 0.0]))
    phase = jp.where(single, ss_phase, stand_phase)
    return State(
        data=data,
        obs_hist=jp.stack([obs, obs, obs]),
        prev_action=jp.zeros(13),
        command=sample_command(k_cmd, scale),
        phase=phase,
        phase_dt=2 * jp.pi * cfg.dt * freq,
        air_time=jp.zeros(2),
        last_contact=jp.zeros(2),
        pos_hist=jp.broadcast_to(base_xy, (VEL_WINDOW, 2)),
        prev_foot_xy=foot_xy,
        rng=rng,
        step=jp.array(0),
        scale=scale,
    )


def observe(state):
    # 137-dim policy input: first 130 dims (3 frames + prev_action) get normalized by
    # the trainer; command and gait phase are appended raw.
    base = jp.concatenate([state.obs_hist.reshape(-1), state.prev_action])
    phase_obs = jp.concatenate([jp.cos(state.phase), jp.sin(state.phase)])
    return jp.concatenate([base, state.command, phase_obs])


def step(model, cfg, state, action):
    ctrl = cfg.home_ctrl.at[10:23].add(action * 0.5)
    ctrl = jp.clip(ctrl, cfg.ctrl_low, cfg.ctrl_high)

    def substep(_, d):
        return mjx.step(model, d.replace(ctrl=ctrl))
    data = jax.lax.fori_loop(0, N_SUBSTEPS, substep, state.data)

    current_contact, self_col = _contacts(data, cfg)
    first_contact = current_contact * (1.0 - state.last_contact)

    # windowed displacement velocity, rotated into the current base frame
    base_xy = data.qpos[:2]
    R = data.xmat[cfg.trunk_id].reshape(3, 3)
    yaw = jp.arctan2(R[1, 0], R[0, 0])
    vel_world = (base_xy - state.pos_hist[0]) / (VEL_WINDOW * cfg.dt)
    lin_vel = jp.array([jp.cos(yaw) * vel_world[0] + jp.sin(yaw) * vel_world[1],
                        -jp.sin(yaw) * vel_world[0] + jp.cos(yaw) * vel_world[1]])
    pos_hist = jp.concatenate([state.pos_hist[1:], base_xy[None]])

    rew, components = reward(data, state.command, lin_vel, action, state.prev_action,
                             state.air_time, first_contact, current_contact,
                             state.phase, state.prev_foot_xy, self_col, cfg)

    foot_xy = jp.array([data.xpos[cfg.left_foot_id, :2], data.xpos[cfg.right_foot_id, :2]])
    air_time = jp.where(current_contact > 0, 0.0, state.air_time + cfg.dt)
    obs = _obs(data, cfg)
    obs_hist = jp.concatenate([state.obs_hist[1:], obs[None]])

    phase = jp.fmod(state.phase + state.phase_dt + jp.pi, 2 * jp.pi) - jp.pi
    phase = jp.where(jp.linalg.norm(state.command) <= 0.01, jp.array([jp.pi, jp.pi]), phase)

    done = terminated(data, cfg)

    # resample command mid-episode so envs that drew small commands still learn to move
    k_cmd, rng = jax.random.split(state.rng)
    resample = (state.step + 1) % RESAMPLE_EVERY == 0
    command = jp.where(resample, sample_command(k_cmd, state.scale), state.command)

    nxt = state.replace(
        data=data, obs_hist=obs_hist, prev_action=action, command=command,
        phase=phase, air_time=air_time, last_contact=current_contact,
        pos_hist=pos_hist, prev_foot_xy=foot_xy, rng=rng, step=state.step + 1,
    )
    # auto-reset on termination; the done flag still flows to GAE
    reset_state = reset(model, cfg, rng, state.scale)
    nxt = jax.tree.map(lambda a, b: jp.where(done, a, b), reset_state, nxt)
    return nxt, rew, done, components
