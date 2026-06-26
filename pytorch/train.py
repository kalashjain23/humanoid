import mujoco
import torch
import numpy as np
from pathlib import Path

from model import Actor, Critic
from running_normalizer import RunningNormalizer
from ppo import PPO
from env import (
    walk_reward_function,
    get_observation,
    check_termination,
    randomize_state,
    get_foot_contacts
)

ROOT = Path(__file__).resolve().parents[1]


def sample_command(scale=1.0):
    """Sample a random velocity command. Forward + turn only (matches goto deployment)."""
    vx = np.random.uniform(0.0, 0.5) * scale    # forward only, includes stand
    vy = 0.0                                     # no lateral motion
    wz = np.random.uniform(-0.5, 0.5) * scale   # full turn range
    return np.array([vx, vy, wz])


def main():
    iterations = 2500 # number of times we will collect rollouts
    checkpoint = 100 # save every n'th checkpoint while training
    resume_from = None # set to a checkpoints/*.pt path to resume
    rollout_length = 1024 # max. number of steps in a rollout
    epoch = 5 # number of epochs a sample
    num_envs = 128 # parallel envs
    minibatch_size = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = mujoco.MjModel.from_xml_path(str(ROOT / "booster_t1/scene.xml"))
    datas = [mujoco.MjData(model) for _ in range(num_envs)]
    obs_histories = [] # maintaining last 3 states
    all_rewards = []

    # one policy step in seconds (5 sim substeps), needed for the gait phase clock
    dt_step = model.opt.timestep * 5

    # ---- precompute reward constants from the clean home keyframe (done once) ----
    mujoco.mj_resetDataKeyframe(model, datas[0], 0)
    mujoco.mj_forward(model, datas[0])
    default_pose = datas[0].qpos[17:30].copy()                      # home pose of 13 ctrl joints
    foot_z0 = np.array([                                            # stance foot height offset
        datas[0].body("left_foot_link").xpos[2],
        datas[0].body("right_foot_link").xpos[2],
    ])

    def _qpos_idx(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid] - 17

    # joint ranges aligned to qpos[17:30]
    jnt_lower = np.zeros(13)
    jnt_upper = np.zeros(13)
    for j in range(model.njnt):
        adr = model.jnt_qposadr[j]
        if 17 <= adr < 30:
            jnt_lower[adr - 17] = model.jnt_range[j, 0]
            jnt_upper[adr - 17] = model.jnt_range[j, 1]

    # pose weights: zero on sagittal walking joints (free to move), 1.0 elsewhere (stay home)
    pose_weights = np.ones(13)
    for nm in ["Left_Hip_Pitch", "Right_Hip_Pitch", "Left_Knee_Pitch",
               "Right_Knee_Pitch", "Left_Ankle_Pitch", "Right_Ankle_Pitch"]:
        pose_weights[_qpos_idx(nm)] = 0.0
    knee_idx = np.array([_qpos_idx("Left_Knee_Pitch"), _qpos_idx("Right_Knee_Pitch")])

    reward_cfg = {
        'default_pose': default_pose,
        'pose_weights': pose_weights,
        'jnt_lower': jnt_lower,
        'jnt_upper': jnt_upper,
        'knee_idx': knee_idx,
        'foot_z0': foot_z0,
        'swing_height': 0.08,
    }
    
    obs_normalizer = RunningNormalizer(130) # normalize only base obs (command appended after)
    actor = Actor()
    critic = Critic()
    ppo = PPO(actor, critic, device=device)

    """
    Linear learning rate scheduler so that policy learns a lot in the beginning
    and does not regress to low reward behaviour in the end of the training
    """
    actor_scheduler = torch.optim.lr_scheduler.LinearLR(
        ppo.actor_opt, start_factor=1.0, end_factor=0.1, total_iters=iterations
    )
    critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        ppo.critic_opt, start_factor=1.0, end_factor=0.1, total_iters=iterations
    )

    # Resume from checkpoint if specified. Loads actor + critic + normalizer + reward history,
    # and (if available) optimizer states + scheduler step count. If optimizer states missing
    # from an older checkpoint, advances scheduler by `iter` steps so LR matches.
    start_iter = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        obs_normalizer.mean = ckpt['obs_mean']
        obs_normalizer.var = ckpt['obs_var']
        obs_normalizer.count = ckpt['obs_count']
        all_rewards = ckpt.get('rewards', [])
        start_iter = ckpt.get('iter', len(all_rewards))
        if 'actor_opt' in ckpt:
            ppo.actor_opt.load_state_dict(ckpt['actor_opt'])
            ppo.critic_opt.load_state_dict(ckpt['critic_opt'])
        if 'actor_sched' in ckpt:
            actor_scheduler.load_state_dict(ckpt['actor_sched'])
            critic_scheduler.load_state_dict(ckpt['critic_sched'])
        else:
            for _ in range(start_iter):
                actor_scheduler.step()
                critic_scheduler.step()
        print(f"resumed from {resume_from} at iter {start_iter}")
    
    ctrl_low = torch.tensor(model.actuator_ctrlrange[:, 0], dtype=torch.float32).to(device)
    ctrl_high = torch.tensor(model.actuator_ctrlrange[:, 1], dtype=torch.float32).to(device)

    for i in range(start_iter + 1, iterations + 1):
        termination_count = 0
        obs_histories = []
        ep_steps = [0] * num_envs
        commands = [] # velocity command per env
        last_actions = [] # previous action per env (used in observation)
        phases = [] # gait phase [phase_L, phase_R] per env
        phase_dts = [] # phase increment per step per env (depends on sampled gait freq)
        command_scale = min(i / 300.0, 1.0)
        resample_every = 250  # resample command mid-rollout to break standing-still attractor
        component_sums = {}
        component_count = 0
        
        for d in datas:
            # reset to randomized initial pose in every env
            randomize_state(model, d)

            first_obs = get_observation(model, d)
            obs_histories.append([first_obs, first_obs, first_obs]) # last 3 states for the networks
            commands.append(sample_command(command_scale))
            last_actions.append(np.zeros(13)) # 13 controllable joints

            # gait clock: feet start antiphase; freq sampled per env
            gait_freq = np.random.uniform(1.25, 1.75)
            phases.append(np.array([0.0, np.pi]))
            phase_dts.append(2 * np.pi * dt_step * gait_freq)
        
        env_rollouts = [[] for _ in range(num_envs)]
        home_ctrl = torch.tensor(datas[0].ctrl.copy(), dtype=torch.float32).to(device)
        
        dt_step = model.opt.timestep * 5  # one policy step in seconds
        air_times = [np.zeros(2) for _ in range(num_envs)]
        last_contacts = [np.zeros(2, dtype=bool) for _ in range(num_envs)]
  
        for step in range(rollout_length):
            # periodically resample commands so envs that drew small commands also learn to move
            if step > 0 and step % resample_every == 0:
                for env_idx in range(num_envs):
                    commands[env_idx] = sample_command(command_scale)

            for env_idx, d in enumerate(datas):
                # concatenate 3-frame history + command to form full observation
                base_obs = np.concatenate(obs_histories[env_idx])
                prev_action = last_actions[env_idx]
                to_normalize = np.concatenate([base_obs, prev_action])       # 130
                norm_part = obs_normalizer.normalize(to_normalize)
                # phase (cos/sin of both feet) appended unnormalized, like the command
                phase_obs = np.concatenate([np.cos(phases[env_idx]), np.sin(phases[env_idx])]) # 4
                full_obs = np.concatenate([norm_part, commands[env_idx], phase_obs]) # 137
                stacked_obs = torch.tensor(full_obs, dtype=torch.float32).to(device)
                
                # calculating actor log probs
                mean = actor(stacked_obs)
                std = actor.log_std.clamp(min=-2.0).exp()
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample().detach()
                log_prob = dist.log_prob(raw_action).sum(-1).detach()
                
                # applying the action on top of home control position
                # indices 10:23 are the waist + leg actuators (indices 0:10 are head + arms)
                action_scaled = home_ctrl.clone()
                action_scaled[10:23] += raw_action * 0.3
                action_scaled = torch.clamp(action_scaled, ctrl_low, ctrl_high)
                d.ctrl[:] = action_scaled.cpu().numpy()
                for _ in range(5):
                    mujoco.mj_step(model, d)
                    
                left_contact, right_contact = get_foot_contacts(model, d)
                current_contact = np.array([left_contact, right_contact])
                first_contact = current_contact & (~last_contacts[env_idx])

                # calculating the reward (reward function) and value (by critic)
                ep_steps[env_idx] += 1
                action_np = raw_action.cpu().numpy()
                reward_val, reward_components = walk_reward_function(
                    model, d, commands[env_idx],
                    air_times[env_idx], first_contact, current_contact,
                    phases[env_idx], reward_cfg,
                )
                reward = torch.tensor(reward_val, dtype=torch.float32)
                for k, v in reward_components.items():
                    component_sums[k] = component_sums.get(k, 0.0) + float(v)
                component_count += 1
                value = critic(stacked_obs).squeeze().detach()
                done = check_termination(d)
                
                air_times[env_idx][current_contact] = 0.0
                air_times[env_idx][~current_contact] += dt_step
                last_contacts[env_idx] = current_contact
                
                # update action history for next step
                last_actions[env_idx] = action_np.copy()
                
                # update observation history
                new_obs = get_observation(model, d)
                obs_histories[env_idx].pop(0)
                obs_histories[env_idx].append(new_obs)

                # advance the gait clock; freeze it when commanded to stand
                p = phases[env_idx] + phase_dts[env_idx]
                p = np.fmod(p + np.pi, 2 * np.pi) - np.pi
                if np.linalg.norm(commands[env_idx]) <= 0.01:
                    p = np.array([np.pi, np.pi])
                phases[env_idx] = p

                if done:
                    # resetting env if terminated
                    randomize_state(model, d)
                    termination_count += 1
                    ep_steps[env_idx] = 0

                    # reset obs history and command
                    first_obs = get_observation(model, d)
                    obs_histories[env_idx] = [first_obs, first_obs, first_obs]
                    commands[env_idx] = sample_command(command_scale)
                    last_actions[env_idx] = np.zeros(13)
                    air_times[env_idx] = np.zeros(2)
                    last_contacts[env_idx] = np.zeros(2, dtype=bool)
                    phases[env_idx] = np.array([0.0, np.pi])
                
                # store: base_obs (for normalizer), full_obs (for network), action, log_prob, reward, value, done
                env_rollouts[env_idx].append((to_normalize.copy(), stacked_obs, raw_action, log_prob, reward, value, done))
                
        all_obs, all_actions, all_log_probs, all_advantages, all_returns, all_rewards_cat = [], [], [], [], [], []
        
        for env_idx in range(num_envs):
            rollout = env_rollouts[env_idx]
            obs = torch.stack([r[1] for r in rollout])
            actions = torch.stack([r[2] for r in rollout])
            log_probs = torch.stack([r[3] for r in rollout])
            rewards = torch.stack([r[4] for r in rollout])
            values = torch.stack([r[5] for r in rollout])
            dones = [r[6] for r in rollout]

            adv, ret = ppo.compute_gae(rewards, values, dones)
            all_obs.append(obs)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_advantages.append(adv)
            all_returns.append(ret)
            all_rewards_cat.append(rewards)
  
        obs_batch = torch.cat(all_obs).to(device)
        actions_batch = torch.cat(all_actions).to(device)
        old_log_probs = torch.cat(all_log_probs).to(device)
        advantages = torch.cat(all_advantages).to(device)
        returns = torch.cat(all_returns).to(device)
        
        # updating the observation normalizer using only the base obs (108 dims, excluding command)
        raw_obs_batch = np.stack([r[0] for rollout in env_rollouts for r in rollout])
        obs_normalizer.update(raw_obs_batch)
  
        # updating actor and critic networks
        ppo.update_networks(obs_batch, actions_batch, old_log_probs, advantages, returns, epoch, minibatch_size)
        
        # linear lr scheduler step
        actor_scheduler.step()
        critic_scheduler.step()
        
        mean_reward = torch.cat(all_rewards_cat).mean().item()
        all_rewards.append(mean_reward)
        comp_str = " ".join(f"{k}={component_sums[k]/component_count:+.2f}" for k in component_sums)
        print(f"iter {i}, mean_reward={mean_reward:.3f}, deaths={termination_count} | {comp_str}")
        
        if i % checkpoint == 0:
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'actor_opt': ppo.actor_opt.state_dict(),
                'critic_opt': ppo.critic_opt.state_dict(),
                'actor_sched': actor_scheduler.state_dict(),
                'critic_sched': critic_scheduler.state_dict(),
                'rewards': all_rewards,
                'iter': i,
                'obs_mean': obs_normalizer.mean,
                'obs_var': obs_normalizer.var,
                'obs_count': obs_normalizer.count,
            }, ROOT / f"checkpoints/checkpoint_{i}.pt")


if __name__ == "__main__":
    main()
