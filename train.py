import mujoco
import torch
from torch import nn
import numpy as np
from torch.optim.adam import Adam

from model import Actor, Critic
from running_normalizer import RunningNormalizer
from reward import (
    upright,
    height_reward,
    control_cost,
)


def get_observation(model, data):
    sensors_data = np.array([])
    for i in range(model.nsensor):
        start = model.sensor_adr[i]
        end = model.sensor_adr[i]+model.sensor_dim[i]
        sensors_data = np.append(sensors_data, data.sensordata[start:end])
        
    obs = np.concatenate([sensors_data, data.qpos[17:30], data.qvel[16:29]])
    
    return obs

def reward_function(data, step):
    stand_reward = height_reward(data) * upright(data) * control_cost(data)
    survival_bonus = min(step / 1000.0, 1.0)  # increases from 0 to 1 over 1000 steps to promote being alive

    return (
        5.0 * stand_reward
    ) * (1.0 + survival_bonus)

def check_termination(data):
    trunk_z = data.body("Trunk").xpos[2]
    trunk_upright = data.body("Trunk").xmat[8]
    return trunk_z < 0.45 or trunk_upright < 0.7

def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    device = values.device
    dtype = values.dtype

    advantages = torch.zeros(T, device=device, dtype=dtype)
    gae = torch.zeros((), device=device, dtype=dtype)
    
    # calculating GAE for the whole rollout
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else torch.zeros((), device=device, dtype=dtype)
        next_nonterminal = torch.tensor(1.0 - float(dones[t]), device=device, dtype=dtype)
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
        
    returns = advantages + values
    return advantages, returns


def update_networks(
    actor: nn.Module,
    actor_opt: torch.optim.Optimizer,
    critic: nn.Module,
    critic_opt: torch.optim.Optimizer,
    obs_batch,
    actions_batch,
    old_log_probs,
    advantages,
    returns,
    e: float,
    epoch: int,
    minibatch_size: int,
):
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    batch_size = obs_batch.shape[0]
    minibatch_size = minibatch_size
    
    for _ in range(epoch):
        indices = torch.randperm(batch_size)
        
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start:start+minibatch_size]
            mb_obs = obs_batch[idx]
            mb_actions = actions_batch[idx]
            mb_old_log_probs = old_log_probs[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]
            
            # get log_probs from the updated actor
            mean = actor(mb_obs)
            std = actor.log_std.clamp(min=-2.0).exp()
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(mb_actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            
            # calculate actor loss
            ratio = (new_log_probs - mb_old_log_probs).exp()
            actor_loss = -torch.minimum(ratio * mb_advantages, torch.clamp(ratio, 1-e, 1+e) * mb_advantages).mean()
            actor_loss = actor_loss - 0.01 * entropy
            
            # calculate critic loss
            values = critic(mb_obs).squeeze()
            critic_loss = nn.functional.mse_loss(values, mb_returns)
            
            # update the networks
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()
    

def main():
    iterations = 200
    checkpoint = 25
    rollout_length = 512
    gamma = 0.99
    lam = 0.95
    e = 0.2
    epoch = 5
    actor_lr = 1e-4
    critic_lr = 1e-4
    num_envs = 128
    minibatch_size = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = mujoco.MjModel.from_xml_path("booster_t1/scene.xml")
    datas = [mujoco.MjData(model) for _ in range(num_envs)]
    obs_histories = []
    all_rewards = []
    
    obs_normalizer = RunningNormalizer(108)
    actor = Actor().to(device)
    critic = Critic().to(device)

    critic_opt = Adam(critic.parameters(), lr=critic_lr)
    actor_opt = Adam(actor.parameters(), lr=actor_lr)
    actor_scheduler = torch.optim.lr_scheduler.LinearLR(
      actor_opt, start_factor=1.0, end_factor=0.0, total_iters=iterations
    )
    critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        critic_opt, start_factor=1.0, end_factor=0.0, total_iters=iterations
    )
    
    ctrl_low = torch.tensor(model.actuator_ctrlrange[:, 0], dtype=torch.float32).to(device)
    ctrl_high = torch.tensor(model.actuator_ctrlrange[:, 1], dtype=torch.float32).to(device)

    for i in range(1, iterations+1):
        termination_count = 0
        obs_histories = []
        ep_steps = [0] * num_envs
        
        for d in datas:
            # reset to initial pose
            mujoco.mj_resetDataKeyframe(model, d, 0)
            mujoco.mj_forward(model, d)

            first_obs = get_observation(model, d)
            obs_histories.append([first_obs, first_obs, first_obs]) # 108 dims for the networks
        
        env_rollouts = [[] for _ in range(num_envs)]
        home_ctrl = torch.tensor(datas[0].ctrl.copy(), dtype=torch.float32).to(device)
        
        for _ in range(rollout_length):
            for env_idx, d in enumerate(datas):
                raw_obs = np.concatenate(obs_histories[env_idx])
                norm_obs = obs_normalizer.normalize(raw_obs)
                stacked_obs = torch.tensor(norm_obs, dtype=torch.float32).to(device)
                
                # calculating actor log probs
                mean = actor(stacked_obs)
                std = actor.log_std.clamp(min=-2.0).exp()
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample().detach()
                log_prob = dist.log_prob(raw_action).sum(-1).detach()
                
                # applying the action on top of home control position
                action_scaled = home_ctrl.clone()
                action_scaled[10:23] += raw_action * 0.3
                action_scaled = torch.clamp(action_scaled, ctrl_low, ctrl_high)
                d.ctrl[:] = action_scaled.cpu().numpy()
                for _ in range(5):
                    mujoco.mj_step(model, d)
                
                # calculating the reward (reward function) and value (by critic)
                ep_steps[env_idx] += 1
                reward = torch.tensor(reward_function(d, ep_steps[env_idx]), dtype=torch.float32)
                value = critic(stacked_obs).squeeze().detach()
                done = check_termination(d)
                
                # update history
                new_obs = get_observation(model, d)
                obs_histories[env_idx].pop(0)
                obs_histories[env_idx].append(new_obs)
                
                if done:
                    mujoco.mj_resetDataKeyframe(model, d, 0)
                    mujoco.mj_forward(model, d)
                    termination_count += 1
                    ep_steps[env_idx] = 0

                    # reset obs history
                    first_obs = get_observation(model, d)
                    obs_histories[env_idx] = [first_obs, first_obs, first_obs]
                
                env_rollouts[env_idx].append((raw_obs.copy(), stacked_obs, raw_action, log_prob, reward, value, done))
                
        all_obs, all_actions, all_log_probs, all_advantages, all_returns, all_rewards_cat = [], [], [], [], [], []
        
        for env_idx in range(num_envs):
            rollout = env_rollouts[env_idx]
            obs = torch.stack([r[1] for r in rollout])
            actions = torch.stack([r[2] for r in rollout])
            log_probs = torch.stack([r[3] for r in rollout])
            rewards = torch.stack([r[4] for r in rollout])
            values = torch.stack([r[5] for r in rollout])
            dones = [r[6] for r in rollout]

            adv, ret = compute_gae(rewards, values, dones, gamma, lam)
            all_obs.append(obs)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_advantages.append(adv)
            all_returns.append(ret)
            all_rewards_cat.append(rewards)
            
        raw_obs_batch = np.stack([r[0] for rollout in env_rollouts for r in rollout])
        obs_normalizer.update(raw_obs_batch)
  
        obs_batch = torch.cat(all_obs).to(device)
        actions_batch = torch.cat(all_actions).to(device)
        old_log_probs = torch.cat(all_log_probs).to(device)
        advantages = torch.cat(all_advantages).to(device)
        returns = torch.cat(all_returns).to(device)
  
        update_networks(actor, actor_opt, critic, critic_opt, obs_batch, actions_batch, old_log_probs, advantages, returns, e, epoch, minibatch_size)
        
        actor_scheduler.step()
        critic_scheduler.step()
        
        mean_reward = torch.cat(all_rewards_cat).mean().item()
        all_rewards.append(mean_reward)
        print(f"iter {i}, mean_reward={mean_reward:.3f}, deaths={termination_count}")
        
        if i % checkpoint == 0:
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'rewards': all_rewards,
                'obs_mean': obs_normalizer.mean,
                'obs_var': obs_normalizer.var,
                'obs_count': obs_normalizer.count,
            }, f'checkpoint_{i}.pt')


if __name__ == "__main__":
    main()
