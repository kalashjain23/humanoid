"""GPU-vectorized PPO training in MJX. Saves a checkpoint export_policy.py converts.

    uv run modal run mjx/train_mjx.py
"""
import modal

GPU = "H100"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("jax[cuda12]", "mujoco", "mujoco-mjx", "flax", "optax", "numpy")
    .add_local_dir("booster_t1", "/root/booster_t1")
    .add_local_file("mjx/mjx_env.py", "/root/mjx_env.py")
    .add_local_file("mjx/ppo_jax.py", "/root/ppo_jax.py")
    .add_local_file("checkpoints/mjx_ft4.pkl", "/root/resume.pkl")
)

app = modal.App("t1-mjx-train", image=image)

# persistent checkpoint storage: periodic saves survive even if the run is killed
ckpt_vol = modal.Volume.from_name("t1-mjx-ckpts", create_if_missing=True)
SAVE_EVERY = 20


def _normalize_stats(mean, var, count, batch):
    bmean, bvar, bcount = batch.mean(0), batch.var(0), batch.shape[0]
    delta = bmean - mean
    tot = count + bcount
    new_mean = mean + delta * bcount / tot
    m2 = var * count + bvar * bcount + delta ** 2 * count * bcount / tot
    return new_mean, m2 / tot, tot


@app.function(gpu=GPU, timeout=2700, volumes={"/ckpt": ckpt_vol})
def train(iterations=3, num_envs=64, rollout_length=128, epochs=5,
          minibatch_size=1024, seed=0, resume=False, lr_init=1e-4, lr_final=1e-5):
    import time
    import pickle
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax
    from functools import partial
    import mjx_env as E
    import ppo_jax as P

    print("jax devices:", jax.devices())
    model, cfg = E.load()

    lr_sched = optax.linear_schedule(lr_init, lr_final, iterations)
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    actor, critic = P.init_states(k, 137, lr_sched)

    nmean = jp.zeros(130)
    nvar = jp.ones(130)
    ncount = jp.array(1e-4)

    if resume:
        import pickle
        with open("/root/resume.pkl", "rb") as f:
            prev = pickle.load(f)
        actor = actor.replace(params=jax.tree.map(jp.asarray, prev["actor_params"]))
        if "critic_params" in prev:
            critic = critic.replace(params=jax.tree.map(jp.asarray, prev["critic_params"]))
        nmean = jp.asarray(prev["obs_mean"])
        nvar = jp.asarray(prev["obs_var"])
        ncount = jp.asarray(prev["obs_count"])
        print("warm-started actor + critic + normalizer from resume.pkl")

    def normalize(o, nmean, nvar):
        base = (o[:130] - nmean) / jp.sqrt(nvar + 1e-8)
        return jp.concatenate([base, o[130:]])

    @partial(jax.jit, static_argnames=("length",))
    def rollout(actor, critic, states, nmean, nvar, rng, length):
        def stepf(carry, _):
            states, rng = carry
            rng, k = jax.random.split(rng)
            obs = jax.vmap(E.observe)(states)
            nobs = jax.vmap(lambda o: normalize(o, nmean, nvar))(obs)
            keys = jax.random.split(k, obs.shape[0])
            actions, logps = jax.vmap(lambda o, kk: P.act(actor, o, kk))(nobs, keys)
            values = jax.vmap(lambda o: P.value(critic, o))(nobs)
            nstates, rewards, dones, comps = jax.vmap(lambda s, a: E.step(model, cfg, s, a))(states, actions)
            return (nstates, rng), (nobs, obs, actions, logps, rewards, values, dones, comps)
        (states, _), traj = jax.lax.scan(stepf, (states, rng), None, length=length)
        return states, traj

    n_minibatch = (rollout_length * num_envs) // minibatch_size
    flat = lambda x: x.reshape(-1, *x.shape[2:])

    to_np = lambda t: jax.tree.map(lambda x: np.asarray(x), t)
    reward_hist = []   # mean reward per iteration, for the training curve
    def checkpoint():
        return {
            "actor_params": to_np(actor.params),
            "critic_params": to_np(critic.params),
            "obs_mean": np.asarray(nmean),
            "obs_var": np.asarray(nvar),
            "obs_count": float(ncount),
            "rewards": list(reward_hist),
        }

    # reset once; step() auto-resets terminated envs, so states persist across iterations
    rng, kr = jax.random.split(rng)
    states = jax.vmap(lambda kk: E.reset(model, cfg, kk, jp.array(0.0)))(jax.random.split(kr, num_envs))

    for i in range(1, iterations + 1):
        scale = min(i / max(1, iterations // 3), 1.0)   # curriculum completes in first third
        states = states.replace(scale=jp.full((num_envs,), scale))
        rng, kr, ku = jax.random.split(rng, 3)

        t0 = time.time()
        states, traj = rollout(actor, critic, states, nmean, nvar, kr, rollout_length)
        nobs, obs, actions, logps, rewards, values, dones, comps = traj
        adv, ret = P.compute_gae(rewards, values, dones)

        nmean, nvar, ncount = _normalize_stats(nmean, nvar, ncount, obs[:, :, :130].reshape(-1, 130))
        actor, critic, al, cl = P.update(
            actor, critic, flat(nobs), flat(actions), flat(logps),
            adv.reshape(-1), ret.reshape(-1), ku, epochs, n_minibatch)

        jax.block_until_ready(rewards)
        dt = time.time() - t0
        sps = rollout_length * num_envs / dt
        cm = {k: float(v.mean()) for k, v in comps.items()}
        reward_hist.append(float(rewards.mean()))
        print(f"iter {i:4d}  reward {float(rewards.mean()):+.3f}  scale {scale:.2f}  "
              f"{sps:,.0f} steps/s | "
              f"track_lin {cm['track_lin']:.2f} track_ang {cm['track_ang']:.2f} "
              f"air {cm['feet_air']:+.2f} phase {cm['feet_phase']:.2f} "
              f"slip {cm['feet_slip']:+.2f} orient {cm['orient']:+.2f} "
              f"arate {cm['action_rate']:+.2f} pose {cm['pose']:+.2f} alive {cm['alive']:.2f}")

        # periodic save to the volume so a killed/crashed run is still recoverable
        if i % SAVE_EVERY == 0 or i == iterations:
            try:
                with open("/ckpt/latest.pkl", "wb") as f:
                    pickle.dump(checkpoint(), f)
                ckpt_vol.commit()
                print(f"  checkpoint saved to volume at iter {i}")
            except Exception as e:
                print(f"  volume save failed at iter {i}: {e}")

    return {
        "actor_params": to_np(actor.params),
        "critic_params": to_np(critic.params),
        "obs_mean": np.asarray(nmean),
        "obs_var": np.asarray(nvar),
        "obs_count": float(ncount),
        "rewards": list(reward_hist),
    }


@app.local_entrypoint()
def main():
    import pickle
    from pathlib import Path

    # warm-start fine-tune from resume.pkl; gentle lr so we refine the gait, not wreck it
    ckpt = train.remote(iterations=150, num_envs=8192, rollout_length=128,
                        epochs=8, minibatch_size=16384, resume=True,
                        lr_init=3e-5, lr_final=1e-5)
    out = Path(__file__).resolve().parents[1] / "checkpoints/mjx_ft5.pkl"
    with open(out, "wb") as f:
        pickle.dump(ckpt, f)
    print("saved", out)
