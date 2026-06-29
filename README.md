# Velocity Command Controlled Humanoid Locomotion

A velocity-command walking policy for the [Booster T1](https://www.boostertech.com/) humanoid in
MuJoCo. A small high-level controller sits on top of the walking policy and steers it toward
target positions.

Robot assets are from [mujoco-menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/booster_t1).

![demo](docs/demo.gif)

PPO is implemented from scratch. I first built and trained it in PyTorch on CPU (`pytorch/`), then
ported the same env and reward to [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) for
GPU-vectorized training (`mjx/`) - thousands of environments in parallel, the full policy in well
under an hour. The shipped policy is MJX-trained and deployed back through the PyTorch inference path.

## What it does

- **Low-level policy** maps proprioception + a velocity command `[vx, vy, wz]` (forward, lateral,
  yaw-rate, all in the robot's base frame) to joint targets for the 13 controllable joints (waist +
  both legs). The arms and head are held at a fixed home pose.
- **High-level controller** (`goto.py`) is a proportional pose controller: it turns a world-frame
  target `(x, y)` into a velocity command and chains a list of waypoints into a path.

## Observation (137-dim)

A history-stacked, partially-normalized proprioceptive vector:

| Block | Dims | Notes |
| --- | --- | --- |
| 3-frame stack of `[IMU, qpos, qvel, projected gravity]` | 117 | 39 per frame; gives the policy velocity/acceleration context |
| previous action | 13 | smoother control, helps sim-to-real |
| velocity command `[vx, vy, wz]` | 3 | appended unnormalized |
| gait phase `[cos pL, cos pR, sin pL, sin pR]` | 4 | appended unnormalized |

The 130-dim proprioception + previous-action block is normalized with a running mean/std; the
command and gait phase are concatenated raw. Projected gravity (world gravity rotated into the trunk
frame) gives the policy an orientation signal without leaning on an absolute world reference, which
is the usual trick for keeping things transferable to hardware.

## Action

The policy outputs a 13-dim delta that gets scaled and added on top of the home joint configuration,
then clamped to the actuator limits. Acting as a residual on a sane default pose keeps exploration
well-behaved from step one - the last layer is zero-initialized, so the policy starts by just holding
the home pose and learns to deviate from there.

## Reward

A few positive task terms plus a handful of small shaping costs, started from the
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) T1 joystick recipe and then
tuned over many runs (see below). Linear velocity is scored as **displacement over a 50-step window**
rather than instantaneous velocity, and both tracking terms are **gated by uprightness** so the
policy cannot bank reward while toppling.

| Term | Weight | Purpose |
| --- | --- | --- |
| `track_lin` | +2.0 | base-frame linear velocity tracking, `exp(-err / 0.25)` |
| `track_ang` | +2.5 | base-frame yaw-rate tracking, `exp(-err / 0.1)` |
| `feet_phase` | +1.0 | track a gait-clock target foot height (the key rhythmic-stepping signal) |
| `feet_air` | +4.0 | reward sustained swing phases, discourage shuffling |
| `alive` | +0.25 | survival bonus |
| `orient` | -1.0 | penalize trunk tilt |
| `ang_vel_xy` | -0.15 | penalize base roll/pitch rates |
| `feet_slip` | -1.0 | penalize horizontal foot speed while in contact |
| `action_rate` | -0.05 | penalize jerky action changes |
| `pose` | -1.0 | keep the non-walking joints near home (walking joints are left free) |
| `dof_lim` | -1.0 | soft joint-limit avoidance |
| `collision` | -1.0 | penalize the feet touching each other |

**The gait-phase clock is the piece that makes it walk.** A per-foot phase advances at a fixed
frequency with the two feet held in antiphase. That phase drives a cubic-bezier target foot-height
profile (`get_rz`), and `feet_phase` rewards each foot for matching it. This bakes in a rhythmic
stepping prior that is hard to discover from velocity tracking alone. The clock freezes when the
commanded velocity is roughly zero, so a "stand" command makes it stand still instead of jogging in
place. Resets use reference-state initialization: half of episodes start in single support with one
foot already lifted to its swing apex, which seeds states a standing reset never reaches.

## What did not work, and how I got here

Most of the work was diagnosis - watching the rollouts and reading the per-term reward logs to tell
a real problem apart from a local optimum. A few that taught me the most:

**Velocity tracking gets gamed by sway.** A plain `exp(-error)` on base velocity is easy to satisfy
by rocking the trunk back and forth without actually translating, and the reward log looks healthy
while the robot goes nowhere - watching the rollout is what made the cheat obvious. Penalizing
vertical and roll/pitch rates only dented it; scoring net displacement over a window instead of
instantaneous velocity removed it, because displacement is genuinely hard to fake.

**A converged local optimum looks like undertraining until you read the logs.** One run settled into
a forward shuffle that barely turned and fell often, and the instinct was to train it longer. The
per-term reward values had been flat for hundreds of iterations, though, which says "converged," not
"still climbing" - so the fix had to be in the reward, not the compute. A tighter yaw kernel, equal
weight on turning, and a slower command curriculum (balance first, full speed later) got it walking.

**Turning had to be in the training distribution.** The policy tracked forward velocity well but
would not come around to a target placed behind it. Reading back the commands it had actually trained
on, they were all forward motion, so the in-place turn the goto controller asks for was something it
had never practiced. Broadening the command sampler to include `vx = 0` turns, and gating the
velocity reward on staying upright, brought the turning in without trading away stability.

Throughout, I changed one reward term at a time - tightening the yaw kernel, gating tracking on
uprightness, making fall termination stricter - and re-checked each change in sim before trusting it,
so anything that hurt the gait showed up immediately instead of being buried in a long run.

## PPO

Implemented from scratch (`pytorch/ppo.py`, `mjx/ppo_jax.py`):

- Generalized Advantage Estimation, `gamma=0.99`, `lambda=0.95`
- Clipped surrogate objective, `epsilon=0.2`, advantages normalized per batch
- A learnable per-dimension log-std, so the policy controls its own exploration
- Entropy bonus and gradient-norm clipping `0.5`
- Separate actor and critic MLPs, Adam

The PyTorch trainer steps 128 environments sequentially on CPU; the MJX trainer runs 8192
environments vectorized on a single GPU (~190k env-steps/s on an H100). Both reset every env to a
randomized pose so the policy does not overfit to one starting configuration.

## Running it

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# train (PyTorch, CPU)
uv run pytorch/train.py

# train (MJX, GPU - run on a CUDA box, e.g. via Modal)
uv run modal run mjx/train_mjx.py

# watch it walk a sequence of waypoints
mjpython pytorch/goto.py
```
