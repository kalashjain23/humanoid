# Velocity Command Controlled Humanoid Locomotion

A velocity command walking policy for the [Booster T1](https://www.boostertech.com/) humanoid in
MuJoCo. A small high-level controller sits on top of the walking policy and steers it toward
target positions.

Robot assets are from [mujoco-menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/booster_t1).

![demo](docs/demo.gif)

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

The policy outputs a 13-dim delta, bounded by `tanh`, that gets scaled and added on top of the home
joint configuration, then clamped to the actuator limits. Acting as a residual on a sane default
pose keeps exploration well-behaved from step one - the last layer is zero-initialized, so the
policy starts by just holding the home pose and learns to deviate from there.

## Reward

The reward follows the [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) T1
joystick recipe: a few positive task terms plus a handful of small shaping costs. I switched to this
after several rounds of hand-tuned reward shaping kept settling into local optima (chatter, foot
crossover, marching in place). It is the same robot and the same task, so the recipe ports cleanly.

| Term | Weight | Purpose |
| --- | --- | --- |
| `track_lin` | +1.0 | base-frame linear velocity tracking, `exp(-err / 0.15)` |
| `track_ang` | +1.0 | base-frame yaw-rate tracking, `exp(-err / 0.25)` |
| `feet_phase` | +1.0 | track a gait-clock target foot height (the key rhythmic-stepping signal) |
| `air_time` | +2.0 | reward sustained swing phases, discourage shuffling |
| `alive` | +0.25 | survival bonus |
| `orient` | -1.0 | penalize trunk tilt |
| `ang_vel_xy` | -0.15 | penalize base roll/pitch rates |
| `feet_dist` | -1.0 | keep lateral foot separation (stops the feet crossing over) |
| `feet_slip` | -0.25 | penalize horizontal foot speed while in contact |
| `pose` | -1.0 | keep the non-walking joints near home (walking joints are left free) |
| `dof_lim` | -1.0 | soft joint-limit avoidance |
| `collision` | -1.0 | penalize the feet touching each other |

**The gait-phase clock is the piece that makes it walk.** A per-foot phase advances at a fixed
frequency with the two feet held in antiphase. That phase drives a cubic-bezier target foot-height
profile (`get_rz`), and `feet_phase` rewards each foot for matching it. This bakes in a rhythmic
stepping prior that is hard to discover from velocity tracking alone. The clock freezes when the
commanded velocity is roughly zero, so a "stand" command makes it stand still instead of jogging in
place.

## What did not work, and how I got here

I did not arrive at the recipe above in one shot. The short version: most of my early effort went
into hand-tuning reward terms, and most of that effort was spent fighting local optima rather than
discovering them.

The first thing that bit me was velocity tracking getting gamed. A naive `exp(-error)` term on base
velocity is easy to satisfy by swaying the trunk back and forth instead of actually translating, so
the robot would rock in place and still collect reward. I tried penalizing vertical velocity and
roll/pitch rates, which helped a little but did not kill the behavior. I then filtered the world
velocity with an EMA before scoring it, which traded sway for a lag in credit assignment. What
finally removed the exploit was scoring displacement over a ~100-step window instead of instantaneous
velocity, which is genuinely hard to fake but added its own latency. Once the gait clock was doing
the heavy lifting I was able to go back to plain instantaneous base-frame velocity, which is what the
final reward uses.

The tolerance on the tracking kernel mattered more than I expected. A loose sigma (`0.25`) makes
standing still look almost as good as tracking, so the policy parks at zero velocity; tightening the
linear term to `0.15` removed that attractor. I kept the yaw sigma loose on purpose, since a tight
sigma flattens the gradient when the heading error is large and the robot stops bothering to turn.

The recurring failure mode through all of this was the policy converging to something that was
locally fine but globally wrong: high-frequency foot chatter, the feet crossing over each other,
and marching in place without going anywhere. Each one needed its own shaping term, and every new
term risked introducing the next local optimum. After enough rounds of this I ported the MuJoCo
Playground T1 joystick reward wholesale, on the logic that it is the same robot and the same task,
so someone had already paid the tuning cost.

Even with the ported reward, one run converged to a forward shuffle: it moved forward but barely
turned, the air-time term had gone negative (short, shuffling steps), and it fell often. I almost
read this as undertraining and threw more compute at it. The component-reward log said otherwise -
the individual terms had been flat for hundreds of iterations, so it was a converged local optimum,
not a policy still on its way up. The fix was the sigma split above plus giving yaw equal weight,
and slowing the command curriculum so the policy learns to stand and balance before it is asked to
hit full commanded speed. That is the run in the demo.

## PPO

Implemented in `ppo.py`:

- Generalized Advantage Estimation, `gamma=0.99`, `lambda=0.95`
- Clipped surrogate objective, `epsilon=0.2`, advantages normalized per batch
- A learnable per-dimension log-std, so the policy controls its own exploration
- Entropy bonus `0.01`, gradient-norm clipping `0.5`
- Separate actor and critic MLPs, both trained with Adam at `1e-4`

Rollouts come from 128 parallel MuJoCo environments, each reset to a randomized pose (trunk tilt,
height, joint angles) so the policy does not overfit to one starting configuration.

## Running it

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# train
uv run pytorch/train.py

# watch it walk a sequence of waypoints
mjpython pytorch/goto.py
```
